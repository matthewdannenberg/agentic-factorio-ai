"""
world/knowledge.py

KnowledgeBase — the agent's persistent, file-backed store of learned game knowledge.

Design
------
The agent starts knowing nothing about specific game content. As it observes the world
(entities on the map, resources in patches, fluids in pipes, recipes crafted, techs
unlocked), it queries Factorio for prototype data and stores the results in CSV/JSON
files under data/knowledge/. On the next run those files are read at startup, so
accumulated knowledge is retained across sessions.

Five knowledge domains, five files:

    data/knowledge/entities.csv   — placed buildings (has unit_number in-game)
    data/knowledge/resources.csv  — mineable resource patches
    data/knowledge/fluids.csv     — fluid prototypes (per temperature variant)
    data/knowledge/recipes.json   — crafting recipes (ingredients, products, time)
    data/knowledge/tech_tree.json — research technologies (prerequisites, unlocks)

File-not-found is not an error — missing files are created empty on first write.

Discovery flow
--------------
When the bridge scanner encounters an unknown entity/resource/fluid name, or when
the research section reveals an unknown tech, the relevant registry's ensure() method
is called. If the name is not in the in-memory store:

  1. query_fn is called with a Lua expression string → returns a raw JSON string
  2. The JSON is parsed into a metadata dataclass
  3. The entry is stored in memory AND appended to the relevant CSV/JSON file

If query_fn is None (unit tests, offline mode), safe placeholder defaults are used
and the entry is still persisted so it can be enriched later.

Architecture constraints
------------------------
- No imports from bridge/, agent/, planning/, llm/, or memory/.
- query_fn is injected by the main loop; this module never calls RCON directly.
- All file I/O is synchronous and intentionally simple — no databases, no ORMs.
- Every public method is safe to call with any string input; nothing raises on
  unknown names.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Type alias for the injected query function.
# Receives a Lua expression, returns raw JSON string (or empty string on failure).
QueryFn = Callable[[str], str]

# ---------------------------------------------------------------------------
# Default data directory — override by passing data_dir to KnowledgeBase()
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIR = Path("data") / "knowledge"


# ===========================================================================
# Entity knowledge
# ===========================================================================

class EntityCategory(Enum):
    SMELTING   = "smelting"
    ASSEMBLY   = "assembly"
    MINING     = "mining"
    POWER      = "power"
    LOGISTICS  = "logistics"
    STORAGE    = "storage"
    DEFENCE    = "defence"
    RESEARCH   = "research"
    TRANSPORT  = "transport"
    OTHER      = "other"


# Mapping from Factorio prototype type strings → EntityCategory.
# Used when learning a new entity from the game — the Lua prototype's .type
# field is translated here. Mod entities with unrecognised types → OTHER.
_PROTO_TYPE_TO_CATEGORY: dict[str, EntityCategory] = {
    "assembling-machine":      EntityCategory.ASSEMBLY,
    "furnace":                 EntityCategory.SMELTING,
    "mining-drill":            EntityCategory.MINING,
    "offshore-pump":           EntityCategory.MINING,
    "boiler":                  EntityCategory.POWER,
    "generator":               EntityCategory.POWER,
    "solar-panel":             EntityCategory.POWER,
    "accumulator":             EntityCategory.POWER,
    "reactor":                 EntityCategory.POWER,
    "heat-interface":          EntityCategory.POWER,
    "electric-pole":           EntityCategory.POWER,
    "inserter":                EntityCategory.LOGISTICS,
    "loader":                  EntityCategory.LOGISTICS,
    "loader-1x1":              EntityCategory.LOGISTICS,
    "pipe":                    EntityCategory.LOGISTICS,
    "pipe-to-ground":          EntityCategory.LOGISTICS,
    "pump":                    EntityCategory.LOGISTICS,
    "roboport":                EntityCategory.LOGISTICS,
    "logistic-container":      EntityCategory.LOGISTICS,
    "container":               EntityCategory.STORAGE,
    "storage-tank":            EntityCategory.STORAGE,
    "transport-belt":          EntityCategory.TRANSPORT,
    "underground-belt":        EntityCategory.TRANSPORT,
    "splitter":                EntityCategory.TRANSPORT,
    "linked-belt":             EntityCategory.TRANSPORT,
    "lane-splitter":           EntityCategory.TRANSPORT,
    "locomotive":              EntityCategory.TRANSPORT,
    "cargo-wagon":             EntityCategory.TRANSPORT,
    "fluid-wagon":             EntityCategory.TRANSPORT,
    "artillery-wagon":         EntityCategory.TRANSPORT,
    "rail":                    EntityCategory.TRANSPORT,
    "straight-rail":           EntityCategory.TRANSPORT,
    "curved-rail":             EntityCategory.TRANSPORT,
    "half-diagonal-rail":      EntityCategory.TRANSPORT,
    "rail-signal":             EntityCategory.TRANSPORT,
    "rail-chain-signal":       EntityCategory.TRANSPORT,
    "train-stop":              EntityCategory.TRANSPORT,
    "turret":                  EntityCategory.DEFENCE,
    "ammo-turret":             EntityCategory.DEFENCE,
    "electric-turret":         EntityCategory.DEFENCE,
    "fluid-turret":            EntityCategory.DEFENCE,
    "artillery-turret":        EntityCategory.DEFENCE,
    "wall":                    EntityCategory.DEFENCE,
    "gate":                    EntityCategory.DEFENCE,
    "land-mine":               EntityCategory.DEFENCE,
    "lab":                     EntityCategory.RESEARCH,
    "rocket-silo":             EntityCategory.ASSEMBLY,   # has recipe slot
    "beacon":                  EntityCategory.OTHER,
    "radar":                   EntityCategory.OTHER,
    "programmable-speaker":    EntityCategory.OTHER,
    "display-panel":           EntityCategory.OTHER,
    "simple-entity-with-owner": EntityCategory.OTHER,
}


@dataclass
class EntityRecord:
    """Everything the agent knows about a placed entity prototype."""
    name: str
    proto_type: str           # Factorio's internal prototype type string
    category: str             # EntityCategory value string (stored as str for CSV compat)
    tile_width: int
    tile_height: int
    has_recipe_slot: bool
    ingredient_slots: int
    output_slots: int
    is_placeholder: bool      # True if learned from defaults (query_fn unavailable)

    @property
    def category_enum(self) -> EntityCategory:
        try:
            return EntityCategory(self.category)
        except ValueError:
            return EntityCategory.OTHER

    # CSV column order
    _FIELDS = [
        "name", "proto_type", "category", "tile_width", "tile_height",
        "has_recipe_slot", "ingredient_slots", "output_slots", "is_placeholder",
    ]

    def to_csv_row(self) -> dict:
        return {
            "name":             self.name,
            "proto_type":       self.proto_type,
            "category":         self.category,
            "tile_width":       self.tile_width,
            "tile_height":      self.tile_height,
            "has_recipe_slot":  self.has_recipe_slot,
            "ingredient_slots": self.ingredient_slots,
            "output_slots":     self.output_slots,
            "is_placeholder":   self.is_placeholder,
        }

    @classmethod
    def from_csv_row(cls, row: dict) -> "EntityRecord":
        return cls(
            name=row["name"],
            proto_type=row.get("proto_type", "unknown"),
            category=row.get("category", EntityCategory.OTHER.value),
            tile_width=int(row.get("tile_width", 1)),
            tile_height=int(row.get("tile_height", 1)),
            has_recipe_slot=row.get("has_recipe_slot", "False") == "True",
            ingredient_slots=int(row.get("ingredient_slots", 0)),
            output_slots=int(row.get("output_slots", 0)),
            is_placeholder=row.get("is_placeholder", "True") == "True",
        )

    @classmethod
    def placeholder(cls, name: str) -> "EntityRecord":
        return cls(
            name=name,
            proto_type="unknown",
            category=EntityCategory.OTHER.value,
            tile_width=1,
            tile_height=1,
            has_recipe_slot=False,
            ingredient_slots=0,
            output_slots=0,
            is_placeholder=True,
        )

    @classmethod
    def from_prototype_json(cls, name: str, data: dict) -> "EntityRecord":
        """Build from the JSON returned by fa.get_entity_prototype()."""
        proto_type = str(data.get("type", "unknown"))
        category = _PROTO_TYPE_TO_CATEGORY.get(proto_type, EntityCategory.OTHER)
        return cls(
            name=name,
            proto_type=proto_type,
            category=category.value,
            tile_width=int(data.get("tile_width", 1)),
            tile_height=int(data.get("tile_height", 1)),
            has_recipe_slot=bool(data.get("has_recipe_slot", False)),
            ingredient_slots=int(data.get("ingredient_slots", 0)),
            output_slots=int(data.get("output_slots", 0)),
            is_placeholder=False,
        )


# ===========================================================================
# Resource knowledge
# ===========================================================================

@dataclass
class ResourceRecord:
    """A mineable resource patch type (iron-ore, crude-oil, etc.)."""
    name: str
    is_fluid: bool            # True for fluid resources (crude-oil, water geysers)
    is_infinite: bool         # True if the patch never fully depletes
    display_name: str         # Human-readable label
    is_placeholder: bool

    _FIELDS = ["name", "is_fluid", "is_infinite", "display_name", "is_placeholder"]

    def to_csv_row(self) -> dict:
        return {
            "name":         self.name,
            "is_fluid":     self.is_fluid,
            "is_infinite":  self.is_infinite,
            "display_name": self.display_name,
            "is_placeholder": self.is_placeholder,
        }

    @classmethod
    def from_csv_row(cls, row: dict) -> "ResourceRecord":
        return cls(
            name=row["name"],
            is_fluid=row.get("is_fluid", "False") == "True",
            is_infinite=row.get("is_infinite", "False") == "True",
            display_name=row.get("display_name", row["name"]),
            is_placeholder=row.get("is_placeholder", "True") == "True",
        )

    @classmethod
    def placeholder(cls, name: str) -> "ResourceRecord":
        return cls(
            name=name,
            is_fluid=False,
            is_infinite=False,
            display_name=name,
            is_placeholder=True,
        )

    @classmethod
    def from_prototype_json(cls, name: str, data: dict) -> "ResourceRecord":
        """Build from the JSON returned by fa.get_resource_prototype()."""
        return cls(
            name=name,
            is_fluid=bool(data.get("is_fluid", False)),
            is_infinite=bool(data.get("is_infinite", False)),
            display_name=str(data.get("display_name", name)),
            is_placeholder=False,
        )


# ===========================================================================
# Fluid knowledge
# ===========================================================================

@dataclass
class FluidRecord:
    """
    A fluid prototype, optionally at a specific temperature.

    Temperature variants are stored as separate rows because steam@165 and
    steam@500 are genuinely distinct in Factorio's recipe system. The key
    used in the registry is "name@temperature" when temperature is not None,
    or just "name" for the base prototype.

    Fields
    ------
    name            : Factorio internal fluid name (e.g. "steam", "crude-oil").
    temperature     : None for the base prototype; integer °C for a variant.
    default_temperature : The fluid's default/base temperature in °C.
    max_temperature : Maximum temperature this fluid can reach.
    is_fuel         : True if this fluid can be used as fuel.
    fuel_value_mj   : Energy content in MJ (0.0 if not a fuel).
    emissions_multiplier : Pollution multiplier when used as fuel (1.0 default).
    is_placeholder  : True if learned from defaults (query_fn unavailable).
    """
    name: str
    temperature: Optional[int]       # None = base prototype
    default_temperature: int
    max_temperature: int
    is_fuel: bool
    fuel_value_mj: float
    emissions_multiplier: float
    is_placeholder: bool

    @property
    def registry_key(self) -> str:
        """Key used in the in-memory store and as a canonical identifier."""
        if self.temperature is not None:
            return f"{self.name}@{self.temperature}"
        return self.name

    _FIELDS = [
        "name", "temperature", "default_temperature", "max_temperature",
        "is_fuel", "fuel_value_mj", "emissions_multiplier", "is_placeholder",
    ]

    def to_csv_row(self) -> dict:
        return {
            "name":                  self.name,
            "temperature":           "" if self.temperature is None else self.temperature,
            "default_temperature":   self.default_temperature,
            "max_temperature":       self.max_temperature,
            "is_fuel":               self.is_fuel,
            "fuel_value_mj":         self.fuel_value_mj,
            "emissions_multiplier":  self.emissions_multiplier,
            "is_placeholder":        self.is_placeholder,
        }

    @classmethod
    def from_csv_row(cls, row: dict) -> "FluidRecord":
        temp_str = row.get("temperature", "")
        return cls(
            name=row["name"],
            temperature=int(temp_str) if temp_str else None,
            default_temperature=int(row.get("default_temperature", 15)),
            max_temperature=int(row.get("max_temperature", 15)),
            is_fuel=row.get("is_fuel", "False") == "True",
            fuel_value_mj=float(row.get("fuel_value_mj", 0.0)),
            emissions_multiplier=float(row.get("emissions_multiplier", 1.0)),
            is_placeholder=row.get("is_placeholder", "True") == "True",
        )

    @classmethod
    def placeholder(cls, name: str, temperature: Optional[int] = None) -> "FluidRecord":
        return cls(
            name=name,
            temperature=temperature,
            default_temperature=15,
            max_temperature=15,
            is_fuel=False,
            fuel_value_mj=0.0,
            emissions_multiplier=1.0,
            is_placeholder=True,
        )

    @classmethod
    def from_prototype_json(cls, name: str, data: dict,
                            temperature: Optional[int] = None) -> "FluidRecord":
        """Build from the JSON returned by fa.get_fluid_prototype()."""
        fuel_value_j = float(data.get("fuel_value", 0.0))
        return cls(
            name=name,
            temperature=temperature,
            default_temperature=int(data.get("default_temperature", 15)),
            max_temperature=int(data.get("max_temperature", 15)),
            is_fuel=fuel_value_j > 0.0,
            fuel_value_mj=fuel_value_j / 1_000_000.0,
            emissions_multiplier=float(data.get("emissions_multiplier", 1.0)),
            is_placeholder=False,
        )


# ===========================================================================
# Recipe knowledge
# ===========================================================================

@dataclass
class IngredientRecord:
    name: str
    amount: float
    is_fluid: bool = False
    temperature: Optional[int] = None   # for fluid ingredients with a required temp


@dataclass
class ProductRecord:
    name: str
    amount: float
    probability: float = 1.0
    is_fluid: bool = False
    temperature: Optional[int] = None   # temperature of produced fluid, if relevant


@dataclass
class RecipeRecord:
    """A crafting recipe."""
    name: str
    category: str             # Factorio crafting category ("crafting", "smelting", etc.)
    crafting_time: float      # seconds
    ingredients: list[IngredientRecord]
    products: list[ProductRecord]
    made_in: list[str]        # entity names that can execute this recipe
    enabled_by_default: bool  # True for hand-craftable recipes available from the start
    is_placeholder: bool

    @classmethod
    def placeholder(cls, name: str) -> "RecipeRecord":
        return cls(
            name=name,
            category="crafting",
            crafting_time=0.5,
            ingredients=[],
            products=[],
            made_in=[],
            enabled_by_default=False,
            is_placeholder=True,
        )

    @classmethod
    def from_prototype_json(cls, name: str, data: dict) -> "RecipeRecord":
        """Build from the JSON returned by fa.get_recipe_prototype()."""
        ingredients = []
        for ing in data.get("ingredients", []):
            ingredients.append(IngredientRecord(
                name=str(ing.get("name", "")),
                amount=float(ing.get("amount", 1)),
                is_fluid=bool(ing.get("type") == "fluid"),
                temperature=ing.get("temperature"),
            ))
        products = []
        for prod in data.get("products", []):
            # Factorio can give an amount_min/amount_max for probabilistic outputs
            amount = float(prod.get("amount", prod.get("amount_min", 1)))
            products.append(ProductRecord(
                name=str(prod.get("name", "")),
                amount=amount,
                probability=float(prod.get("probability", 1.0)),
                is_fluid=bool(prod.get("type") == "fluid"),
                temperature=prod.get("temperature"),
            ))
        return cls(
            name=name,
            category=str(data.get("category", "crafting")),
            crafting_time=float(data.get("energy_required", 0.5)),
            ingredients=ingredients,
            products=products,
            made_in=list(data.get("made_in", [])),
            enabled_by_default=bool(data.get("enabled", False)),
            is_placeholder=False,
        )

    def to_json_obj(self) -> dict:
        return {
            "name":              self.name,
            "category":          self.category,
            "crafting_time":     self.crafting_time,
            "ingredients":       [
                {"name": i.name, "amount": i.amount,
                 "is_fluid": i.is_fluid, "temperature": i.temperature}
                for i in self.ingredients
            ],
            "products":          [
                {"name": p.name, "amount": p.amount,
                 "probability": p.probability, "is_fluid": p.is_fluid,
                 "temperature": p.temperature}
                for p in self.products
            ],
            "made_in":           self.made_in,
            "enabled_by_default": self.enabled_by_default,
            "is_placeholder":    self.is_placeholder,
        }

    @classmethod
    def from_json_obj(cls, data: dict) -> "RecipeRecord":
        ingredients = [
            IngredientRecord(
                name=i["name"], amount=i["amount"],
                is_fluid=i.get("is_fluid", False),
                temperature=i.get("temperature"),
            ) for i in data.get("ingredients", [])
        ]
        products = [
            ProductRecord(
                name=p["name"], amount=p["amount"],
                probability=p.get("probability", 1.0),
                is_fluid=p.get("is_fluid", False),
                temperature=p.get("temperature"),
            ) for p in data.get("products", [])
        ]
        return cls(
            name=data["name"],
            category=data.get("category", "crafting"),
            crafting_time=float(data.get("crafting_time", 0.5)),
            ingredients=ingredients,
            products=products,
            made_in=data.get("made_in", []),
            enabled_by_default=bool(data.get("enabled_by_default", False)),
            is_placeholder=bool(data.get("is_placeholder", True)),
        )


# ===========================================================================
# Tech tree knowledge
# ===========================================================================

@dataclass
class TechRecord:
    """A research technology node."""
    name: str
    prerequisites: list[str]
    unlocks_recipes: list[str]
    unlocks_entities: list[str]
    requires_dlc: bool
    is_placeholder: bool

    @classmethod
    def placeholder(cls, name: str) -> "TechRecord":
        return cls(
            name=name,
            prerequisites=[],
            unlocks_recipes=[],
            unlocks_entities=[],
            requires_dlc=False,
            is_placeholder=True,
        )

    @classmethod
    def from_prototype_json(cls, name: str, data: dict) -> "TechRecord":
        """Build from the JSON returned by fa.get_technology()."""
        unlocks_recipes = []
        unlocks_entities = []
        for effect in data.get("effects", []):
            etype = effect.get("type", "")
            if etype == "unlock-recipe":
                unlocks_recipes.append(str(effect.get("recipe", "")))
            elif etype == "give-item":
                # Some techs give items (e.g. equipment); treat as entity unlock
                unlocks_entities.append(str(effect.get("item", "")))
        return cls(
            name=name,
            prerequisites=list(data.get("prerequisites", [])),
            unlocks_recipes=unlocks_recipes,
            unlocks_entities=unlocks_entities,
            requires_dlc=bool(data.get("requires_dlc", False)),
            is_placeholder=False,
        )

    def to_json_obj(self) -> dict:
        return {
            "name":              self.name,
            "prerequisites":     self.prerequisites,
            "unlocks_recipes":   self.unlocks_recipes,
            "unlocks_entities":  self.unlocks_entities,
            "requires_dlc":      self.requires_dlc,
            "is_placeholder":    self.is_placeholder,
        }

    @classmethod
    def from_json_obj(cls, data: dict) -> "TechRecord":
        return cls(
            name=data["name"],
            prerequisites=data.get("prerequisites", []),
            unlocks_recipes=data.get("unlocks_recipes", []),
            unlocks_entities=data.get("unlocks_entities", []),
            requires_dlc=bool(data.get("requires_dlc", False)),
            is_placeholder=bool(data.get("is_placeholder", True)),
        )


# ===========================================================================
# KnowledgeBase
# ===========================================================================

class KnowledgeBase:
    """
    The agent's persistent store of learned game knowledge.

    Owns five registries (entities, resources, fluids, recipes, tech_tree).
    Loads from CSV/JSON files at construction; writes back on every new discovery.

    Parameters
    ----------
    data_dir  : Directory for the five knowledge files. Created if absent.
    query_fn  : Callable(lua_expr: str) -> raw_json: str, injected by the main
                loop. When None (offline/test mode), unknowns get placeholders.
    """

    def __init__(
        self,
        data_dir: Path = _DEFAULT_DATA_DIR,
        query_fn: Optional[QueryFn] = None,
    ) -> None:
        self._dir = Path(data_dir)
        self._query_fn = query_fn

        self._dir.mkdir(parents=True, exist_ok=True)

        # In-memory stores
        self._entities:  dict[str, EntityRecord]  = {}
        self._resources: dict[str, ResourceRecord] = {}
        self._fluids:    dict[str, FluidRecord]   = {}
        self._recipes:   dict[str, RecipeRecord]  = {}
        self._techs:     dict[str, TechRecord]    = {}

        self._load_all()

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    def _path(self, filename: str) -> Path:
        return self._dir / filename

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._load_csv("entities.csv",  self._entities,
                       EntityRecord.from_csv_row,  EntityRecord._FIELDS)
        self._load_csv("resources.csv", self._resources,
                       ResourceRecord.from_csv_row, ResourceRecord._FIELDS)
        self._load_csv("fluids.csv",    self._fluids,
                       self._fluid_from_row,        FluidRecord._FIELDS,
                       key_fn=lambda r: r.registry_key)
        self._load_json("recipes.json",   self._recipes,  RecipeRecord.from_json_obj)
        self._load_json("tech_tree.json", self._techs,    TechRecord.from_json_obj)

    def _load_csv(self, filename: str, store: dict,
                  row_fn, fields: list[str],
                  key_fn=None) -> None:
        path = self._path(filename)
        if not path.exists():
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        record = row_fn(row)
                        key = key_fn(record) if key_fn else record.name
                        store[key] = record
                    except Exception as exc:
                        logger.warning("Skipping malformed row in %s: %s — %s",
                                       filename, row, exc)
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)

    def _load_json(self, filename: str, store: dict, record_fn) -> None:
        path = self._path(filename)
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("%s: expected JSON object at top level", path)
                return
            for name, obj in data.items():
                try:
                    obj["name"] = name   # ensure name field present
                    store[name] = record_fn(obj)
                except Exception as exc:
                    logger.warning("Skipping malformed entry '%s' in %s: %s",
                                   name, filename, exc)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s: %s", path, exc)

    @staticmethod
    def _fluid_from_row(row: dict) -> FluidRecord:
        return FluidRecord.from_csv_row(row)

    # ------------------------------------------------------------------
    # Persist (append for CSV; full rewrite for JSON)
    # ------------------------------------------------------------------

    def _append_csv(self, filename: str, record, fields: list[str]) -> None:
        path = self._path(filename)
        write_header = not path.exists() or path.stat().st_size == 0
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                writer.writerow(record.to_csv_row())
        except OSError as exc:
            logger.error("Could not write to %s: %s", path, exc)

    def _rewrite_json(self, filename: str, store: dict, to_obj_fn) -> None:
        path = self._path(filename)
        try:
            data = {name: to_obj_fn(record) for name, record in store.items()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.error("Could not write to %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Lua query helpers
    # ------------------------------------------------------------------

    def _query(self, lua_expr: str) -> Optional[dict]:
        """Send a Lua query and parse the JSON result. Returns None on any failure."""
        if self._query_fn is None:
            return None
        try:
            raw = self._query_fn(lua_expr)
            if not raw:
                return None
            data = json.loads(raw)
            if isinstance(data, dict) and not data.get("ok", True) is False:
                return data
            if isinstance(data, dict) and data.get("ok") is False:
                logger.debug("Lua query failed: %s", data.get("reason"))
                return None
            return data
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("Lua query error for %r: %s", lua_expr, exc)
            return None

    # ------------------------------------------------------------------
    # Entity registry
    # ------------------------------------------------------------------

    def ensure_entity(self, name: str) -> EntityRecord:
        """
        Return the EntityRecord for name, querying Factorio if unknown.
        Never raises. Returns a placeholder if discovery fails.
        """
        if name in self._entities:
            return self._entities[name]

        record = self._discover_entity(name)
        self._entities[name] = record
        self._append_csv("entities.csv", record, EntityRecord._FIELDS)
        return record

    def _discover_entity(self, name: str) -> EntityRecord:
        data = self._query(f'rcon.print(fa.get_entity_prototype("{name}"))')
        if data is None:
            logger.info("Entity %r: query unavailable, using placeholder", name)
            return EntityRecord.placeholder(name)
        try:
            return EntityRecord.from_prototype_json(name, data)
        except Exception as exc:
            logger.warning("Entity %r: failed to parse prototype: %s", name, exc)
            return EntityRecord.placeholder(name)

    def get_entity(self, name: str) -> Optional[EntityRecord]:
        return self._entities.get(name)

    def all_entities(self) -> dict[str, EntityRecord]:
        return dict(self._entities)

    # ------------------------------------------------------------------
    # Resource registry
    # ------------------------------------------------------------------

    def ensure_resource(self, name: str) -> ResourceRecord:
        """Return the ResourceRecord for name, querying Factorio if unknown."""
        if name in self._resources:
            return self._resources[name]

        record = self._discover_resource(name)
        self._resources[name] = record
        self._append_csv("resources.csv", record, ResourceRecord._FIELDS)
        return record

    def _discover_resource(self, name: str) -> ResourceRecord:
        data = self._query(f'rcon.print(fa.get_resource_prototype("{name}"))')
        if data is None:
            logger.info("Resource %r: query unavailable, using placeholder", name)
            return ResourceRecord.placeholder(name)
        try:
            return ResourceRecord.from_prototype_json(name, data)
        except Exception as exc:
            logger.warning("Resource %r: failed to parse prototype: %s", name, exc)
            return ResourceRecord.placeholder(name)

    def get_resource(self, name: str) -> Optional[ResourceRecord]:
        return self._resources.get(name)

    def all_resources(self) -> dict[str, ResourceRecord]:
        return dict(self._resources)

    # ------------------------------------------------------------------
    # Fluid registry
    # ------------------------------------------------------------------

    def ensure_fluid(self, name: str,
                     temperature: Optional[int] = None) -> FluidRecord:
        """
        Return the FluidRecord for name (and optional temperature variant).
        Registry key is "name@temperature" for variants, "name" for base.
        """
        key = f"{name}@{temperature}" if temperature is not None else name
        if key in self._fluids:
            return self._fluids[key]

        record = self._discover_fluid(name, temperature)
        self._fluids[key] = record
        self._append_csv("fluids.csv", record, FluidRecord._FIELDS)
        return record

    def _discover_fluid(self, name: str,
                        temperature: Optional[int]) -> FluidRecord:
        data = self._query(f'rcon.print(fa.get_fluid_prototype("{name}"))')
        if data is None:
            logger.info("Fluid %r (temp=%s): query unavailable, using placeholder",
                        name, temperature)
            return FluidRecord.placeholder(name, temperature)
        try:
            return FluidRecord.from_prototype_json(name, data, temperature)
        except Exception as exc:
            logger.warning("Fluid %r: failed to parse prototype: %s", name, exc)
            return FluidRecord.placeholder(name, temperature)

    def get_fluid(self, name: str,
                  temperature: Optional[int] = None) -> Optional[FluidRecord]:
        key = f"{name}@{temperature}" if temperature is not None else name
        return self._fluids.get(key)

    def all_fluids(self) -> dict[str, FluidRecord]:
        return dict(self._fluids)

    # ------------------------------------------------------------------
    # Recipe registry
    # ------------------------------------------------------------------

    def ensure_recipe(self, name: str) -> RecipeRecord:
        """Return the RecipeRecord for name, querying Factorio if unknown."""
        if name in self._recipes:
            return self._recipes[name]

        record = self._discover_recipe(name)
        self._recipes[name] = record
        self._rewrite_json("recipes.json", self._recipes,
                            lambda r: r.to_json_obj())
        return record

    def _discover_recipe(self, name: str) -> RecipeRecord:
        data = self._query(f'rcon.print(fa.get_recipe_prototype("{name}"))')
        if data is None:
            logger.info("Recipe %r: query unavailable, using placeholder", name)
            return RecipeRecord.placeholder(name)
        try:
            return RecipeRecord.from_prototype_json(name, data)
        except Exception as exc:
            logger.warning("Recipe %r: failed to parse prototype: %s", name, exc)
            return RecipeRecord.placeholder(name)

    def get_recipe(self, name: str) -> Optional[RecipeRecord]:
        return self._recipes.get(name)

    def all_recipes(self) -> dict[str, RecipeRecord]:
        return dict(self._recipes)

    def recipes_for_product(self, product_name: str) -> list[RecipeRecord]:
        """All known recipes that produce the given item or fluid."""
        return [
            r for r in self._recipes.values()
            if any(p.name == product_name for p in r.products)
        ]

    def recipes_for_ingredient(self, ingredient_name: str) -> list[RecipeRecord]:
        """All known recipes that consume the given item or fluid."""
        return [
            r for r in self._recipes.values()
            if any(i.name == ingredient_name for i in r.ingredients)
        ]

    # ------------------------------------------------------------------
    # Tech tree registry
    # ------------------------------------------------------------------

    def ensure_tech(self, name: str) -> TechRecord:
        """Return the TechRecord for name, querying Factorio if unknown."""
        if name in self._techs:
            return self._techs[name]

        record = self._discover_tech(name)
        self._techs[name] = record
        self._rewrite_json("tech_tree.json", self._techs,
                            lambda r: r.to_json_obj())
        return record

    def _discover_tech(self, name: str) -> TechRecord:
        data = self._query(f'rcon.print(fa.get_technology("{name}"))')
        if data is None:
            logger.info("Tech %r: query unavailable, using placeholder", name)
            return TechRecord.placeholder(name)
        try:
            return TechRecord.from_prototype_json(name, data)
        except Exception as exc:
            logger.warning("Tech %r: failed to parse prototype: %s", name, exc)
            return TechRecord.placeholder(name)

    def get_tech(self, name: str) -> Optional[TechRecord]:
        return self._techs.get(name)

    def all_techs(self) -> dict[str, TechRecord]:
        return dict(self._techs)

    def prerequisites(self, name: str) -> list[str]:
        record = self._techs.get(name)
        return list(record.prerequisites) if record else []

    def all_prerequisites(self, name: str) -> set[str]:
        """Full transitive prerequisite set."""
        result: set[str] = set()
        stack = self.prerequisites(name)
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.add(current)
            stack.extend(self.prerequisites(current))
        return result

    # ------------------------------------------------------------------
    # Diagnostic / summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "entities":  len(self._entities),
            "resources": len(self._resources),
            "fluids":    len(self._fluids),
            "recipes":   len(self._recipes),
            "techs":     len(self._techs),
            "data_dir":  str(self._dir),
        }
