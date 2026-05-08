"""
world/knowledge.py

KnowledgeBase — the agent's persistent, SQLite-backed store of learned game knowledge.

Design
------
The agent starts knowing nothing about specific game content. As it observes the world
(entities on the map, resources in patches, fluids in pipes, recipes crafted, techs
unlocked), it queries Factorio for prototype data and stores the results in a SQLite
database at data/knowledge/knowledge.db. On the next run the DB is read at startup,
so accumulated knowledge is retained across sessions.

Five knowledge domains, eleven tables:

    entities                — placed buildings (has unit_number in-game)
    resources               — mineable resource patches
    fluids                  — fluid prototypes (per temperature variant)
    recipes                 — crafting recipe headers
    recipe_ingredients      — normalised ingredient rows (FK → recipes)
    recipe_products         — normalised product rows (FK → recipes)
    recipe_made_in          — entity names that can run each recipe (FK → recipes)
    techs                   — research technology nodes
    tech_prerequisites      — prerequisite edges (FK → techs)
    tech_unlocks_recipes    — recipe unlock effects (FK → techs)
    tech_unlocks_entities   — entity/item unlock effects (FK → techs)

Why SQLite
----------
The planning layer needs cross-domain queries that are expensive as dict scans:
  - "All recipes that use iron-plate as an ingredient"
  - "All recipes produceable in an assembling-machine-2"
  - "Full production chain for processing-unit" (recursive CTE)
  - "Which techs unlock recipes that output Y?"
SQLite handles all of these with a single query and indexed lookups. The stdlib
sqlite3 module requires no dependencies and stores to a single portable file.

Public API
--------------------------------------------------------------------
KnowledgeBase(data_dir, query_fn)
  .ensure_entity/resource/fluid/recipe/tech(name)  → Record
  .get_entity/resource/fluid/recipe/tech(name)      → Record | None
  .all_entities/resources/fluids/recipes/techs()    → dict[str, Record]
  .recipes_for_product(name)       → list[RecipeRecord]
  .recipes_for_ingredient(name)    → list[RecipeRecord]
  .recipes_made_in(entity_name)    → list[RecipeRecord]   
  .techs_unlocking_recipe(name)    → list[TechRecord]     
  .production_chain(item)          → set[str]             
  .prerequisites(name)             → list[str]
  .all_prerequisites(name)         → set[str]
  .summary()                       → dict

Architecture constraints
------------------------
- No imports from bridge/, agent/, planning/, llm/, or memory/.
- query_fn is injected by the main loop; this module never calls RCON directly.
- Every public method is safe to call with any string input; nothing raises on
  unknown names.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

QueryFn = Callable[[str], str]

_DEFAULT_DATA_DIR = Path("data") / "knowledge"
_DB_NAME = "knowledge.db"


# ===========================================================================
# EntityCategory
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


# Minimal mapping: only prototype types where the category meaningfully changes
# how the agent reasons (what actions are valid, what slots exist, etc.).
# Unrecognised types fall through to OTHER safely.
_PROTO_TYPE_TO_CATEGORY: dict[str, EntityCategory] = {
    "assembling-machine": EntityCategory.ASSEMBLY,
    "furnace":            EntityCategory.SMELTING,
    "mining-drill":       EntityCategory.MINING,
    "lab":                EntityCategory.RESEARCH,
    "rocket-silo":        EntityCategory.ASSEMBLY,
    "container":          EntityCategory.STORAGE,
    "logistic-container": EntityCategory.LOGISTICS,
    "ammo-turret":        EntityCategory.DEFENCE,
    "electric-turret":    EntityCategory.DEFENCE,
    "fluid-turret":       EntityCategory.DEFENCE,
    "artillery-turret":   EntityCategory.DEFENCE,
    "wall":               EntityCategory.DEFENCE,
}


# ===========================================================================
# Record dataclasses
# ===========================================================================

@dataclass
class EntityRecord:
    name: str
    proto_type: str
    category: str          # EntityCategory.value string
    tile_width: int
    tile_height: int
    has_recipe_slot: bool
    ingredient_slots: int
    output_slots: int
    is_placeholder: bool

    @property
    def category_enum(self) -> EntityCategory:
        try:
            return EntityCategory(self.category)
        except ValueError:
            return EntityCategory.OTHER

    @classmethod
    def placeholder(cls, name: str) -> "EntityRecord":
        return cls(name=name, proto_type="unknown",
                   category=EntityCategory.OTHER.value,
                   tile_width=1, tile_height=1,
                   has_recipe_slot=False, ingredient_slots=0,
                   output_slots=0, is_placeholder=True)

    @classmethod
    def from_prototype_json(cls, name: str, data: dict) -> "EntityRecord":
        proto_type = str(data.get("type", "unknown"))
        category = _PROTO_TYPE_TO_CATEGORY.get(proto_type, EntityCategory.OTHER)
        return cls(
            name=name, proto_type=proto_type, category=category.value,
            tile_width=int(data.get("tile_width", 1)),
            tile_height=int(data.get("tile_height", 1)),
            has_recipe_slot=bool(data.get("has_recipe_slot", False)),
            ingredient_slots=int(data.get("ingredient_slots", 0)),
            output_slots=int(data.get("output_slots", 0)),
            is_placeholder=False,
        )


@dataclass
class ResourceRecord:
    name: str
    is_fluid: bool
    is_infinite: bool
    display_name: str
    is_placeholder: bool

    @classmethod
    def placeholder(cls, name: str) -> "ResourceRecord":
        return cls(name=name, is_fluid=False, is_infinite=False,
                   display_name=name, is_placeholder=True)

    @classmethod
    def from_prototype_json(cls, name: str, data: dict) -> "ResourceRecord":
        return cls(
            name=name,
            is_fluid=bool(data.get("is_fluid", False)),
            is_infinite=bool(data.get("is_infinite", False)),
            display_name=str(data.get("display_name", name)),
            is_placeholder=False,
        )


@dataclass
class FluidRecord:
    name: str
    temperature: Optional[int]     # None = base prototype
    default_temperature: int
    max_temperature: int
    is_fuel: bool
    fuel_value_mj: float
    emissions_multiplier: float
    is_placeholder: bool

    @property
    def registry_key(self) -> str:
        return f"{self.name}@{self.temperature}" if self.temperature is not None else self.name

    @classmethod
    def placeholder(cls, name: str, temperature: Optional[int] = None) -> "FluidRecord":
        return cls(name=name, temperature=temperature,
                   default_temperature=15, max_temperature=15,
                   is_fuel=False, fuel_value_mj=0.0,
                   emissions_multiplier=1.0, is_placeholder=True)

    @classmethod
    def from_prototype_json(cls, name: str, data: dict,
                            temperature: Optional[int] = None) -> "FluidRecord":
        fuel_j = float(data.get("fuel_value", 0.0))
        return cls(
            name=name, temperature=temperature,
            default_temperature=int(data.get("default_temperature", 15)),
            max_temperature=int(data.get("max_temperature", 15)),
            is_fuel=fuel_j > 0.0,
            fuel_value_mj=fuel_j / 1_000_000.0,
            emissions_multiplier=float(data.get("emissions_multiplier", 1.0)),
            is_placeholder=False,
        )


@dataclass
class IngredientRecord:
    name: str
    amount: float
    is_fluid: bool = False
    temperature: Optional[int] = None


@dataclass
class ProductRecord:
    name: str
    amount: float
    probability: float = 1.0
    is_fluid: bool = False
    temperature: Optional[int] = None


@dataclass
class RecipeRecord:
    name: str
    category: str
    crafting_time: float
    ingredients: list[IngredientRecord]
    products: list[ProductRecord]
    made_in: list[str]
    enabled_by_default: bool
    is_placeholder: bool

    @classmethod
    def placeholder(cls, name: str) -> "RecipeRecord":
        return cls(name=name, category="crafting", crafting_time=0.5,
                   ingredients=[], products=[], made_in=[],
                   enabled_by_default=False, is_placeholder=True)

    @classmethod
    def from_prototype_json(cls, name: str, data: dict) -> "RecipeRecord":
        ingredients = [
            IngredientRecord(
                name=str(i.get("name", "")),
                amount=float(i.get("amount", 1)),
                is_fluid=bool(i.get("type") == "fluid"),
                temperature=i.get("temperature"),
            ) for i in data.get("ingredients", [])
        ]
        products = [
            ProductRecord(
                name=str(p.get("name", "")),
                amount=float(p.get("amount", p.get("amount_min", 1))),
                probability=float(p.get("probability", 1.0)),
                is_fluid=bool(p.get("type") == "fluid"),
                temperature=p.get("temperature"),
            ) for p in data.get("products", [])
        ]
        return cls(
            name=name,
            category=str(data.get("category", "crafting")),
            crafting_time=float(data.get("energy_required", 0.5)),
            ingredients=ingredients, products=products,
            made_in=list(data.get("made_in", [])),
            enabled_by_default=bool(data.get("enabled", False)),
            is_placeholder=False,
        )


@dataclass
class TechRecord:
    name: str
    prerequisites: list[str]
    unlocks_recipes: list[str]
    unlocks_entities: list[str]
    requires_dlc: bool
    is_placeholder: bool

    @classmethod
    def placeholder(cls, name: str) -> "TechRecord":
        return cls(name=name, prerequisites=[], unlocks_recipes=[],
                   unlocks_entities=[], requires_dlc=False, is_placeholder=True)

    @classmethod
    def from_prototype_json(cls, name: str, data: dict) -> "TechRecord":
        unlocks_recipes, unlocks_entities = [], []
        for effect in data.get("effects", []):
            etype = effect.get("type", "")
            if etype == "unlock-recipe":
                unlocks_recipes.append(str(effect.get("recipe", "")))
            elif etype == "give-item":
                unlocks_entities.append(str(effect.get("item", "")))
        return cls(
            name=name,
            prerequisites=list(data.get("prerequisites", [])),
            unlocks_recipes=unlocks_recipes,
            unlocks_entities=unlocks_entities,
            requires_dlc=bool(data.get("requires_dlc", False)),
            is_placeholder=False,
        )


# ===========================================================================
# Schema
# ===========================================================================

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS entities (
    name             TEXT PRIMARY KEY,
    proto_type       TEXT NOT NULL DEFAULT 'unknown',
    category         TEXT NOT NULL DEFAULT 'other',
    tile_width       INTEGER NOT NULL DEFAULT 1,
    tile_height      INTEGER NOT NULL DEFAULT 1,
    has_recipe_slot  INTEGER NOT NULL DEFAULT 0,
    ingredient_slots INTEGER NOT NULL DEFAULT 0,
    output_slots     INTEGER NOT NULL DEFAULT 0,
    is_placeholder   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS resources (
    name           TEXT PRIMARY KEY,
    is_fluid       INTEGER NOT NULL DEFAULT 0,
    is_infinite    INTEGER NOT NULL DEFAULT 0,
    display_name   TEXT NOT NULL DEFAULT '',
    is_placeholder INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS fluids (
    name                 TEXT NOT NULL,
    temperature          INTEGER,
    default_temperature  INTEGER NOT NULL DEFAULT 15,
    max_temperature      INTEGER NOT NULL DEFAULT 15,
    is_fuel              INTEGER NOT NULL DEFAULT 0,
    fuel_value_mj        REAL NOT NULL DEFAULT 0.0,
    emissions_multiplier REAL NOT NULL DEFAULT 1.0,
    is_placeholder       INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (name, temperature)
);

CREATE TABLE IF NOT EXISTS recipes (
    name               TEXT PRIMARY KEY,
    category           TEXT NOT NULL DEFAULT 'crafting',
    crafting_time      REAL NOT NULL DEFAULT 0.5,
    enabled_by_default INTEGER NOT NULL DEFAULT 0,
    is_placeholder     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS recipe_ingredients (
    recipe_name  TEXT NOT NULL REFERENCES recipes(name) ON DELETE CASCADE,
    item_name    TEXT NOT NULL,
    amount       REAL NOT NULL,
    is_fluid     INTEGER NOT NULL DEFAULT 0,
    temperature  INTEGER,
    position     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS recipe_products (
    recipe_name  TEXT NOT NULL REFERENCES recipes(name) ON DELETE CASCADE,
    item_name    TEXT NOT NULL,
    amount       REAL NOT NULL,
    probability  REAL NOT NULL DEFAULT 1.0,
    is_fluid     INTEGER NOT NULL DEFAULT 0,
    temperature  INTEGER,
    position     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS recipe_made_in (
    recipe_name  TEXT NOT NULL REFERENCES recipes(name) ON DELETE CASCADE,
    entity_name  TEXT NOT NULL,
    PRIMARY KEY (recipe_name, entity_name)
);

CREATE TABLE IF NOT EXISTS techs (
    name           TEXT PRIMARY KEY,
    requires_dlc   INTEGER NOT NULL DEFAULT 0,
    is_placeholder INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS tech_prerequisites (
    tech_name  TEXT NOT NULL REFERENCES techs(name) ON DELETE CASCADE,
    prereq     TEXT NOT NULL,
    PRIMARY KEY (tech_name, prereq)
);

CREATE TABLE IF NOT EXISTS tech_unlocks_recipes (
    tech_name    TEXT NOT NULL REFERENCES techs(name) ON DELETE CASCADE,
    recipe_name  TEXT NOT NULL,
    PRIMARY KEY (tech_name, recipe_name)
);

CREATE TABLE IF NOT EXISTS tech_unlocks_entities (
    tech_name    TEXT NOT NULL REFERENCES techs(name) ON DELETE CASCADE,
    entity_name  TEXT NOT NULL,
    PRIMARY KEY (tech_name, entity_name)
);

CREATE INDEX IF NOT EXISTS idx_recipe_ingredients_item ON recipe_ingredients(item_name);
CREATE INDEX IF NOT EXISTS idx_recipe_products_item    ON recipe_products(item_name);
CREATE INDEX IF NOT EXISTS idx_recipe_made_in_entity   ON recipe_made_in(entity_name);
CREATE INDEX IF NOT EXISTS idx_tech_unlocks_recipe     ON tech_unlocks_recipes(recipe_name);
CREATE INDEX IF NOT EXISTS idx_tech_prereq_prereq      ON tech_prerequisites(prereq);
"""


# ===========================================================================
# KnowledgeBase
# ===========================================================================

class KnowledgeBase:
    """
    The agent's persistent store of learned game knowledge, backed by SQLite.

    In-memory dicts act as a write-through cache: reads always hit memory,
    writes go to both memory and the DB atomically. SQLite is opened once per
    KnowledgeBase lifetime.

    Parameters
    ----------
    data_dir  : Directory containing knowledge.db. Created if absent.
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

        db_path = self._dir / _DB_NAME
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        self._entities:  dict[str, EntityRecord]  = {}
        self._resources: dict[str, ResourceRecord] = {}
        self._fluids:    dict[str, FluidRecord]    = {}
        self._recipes:   dict[str, RecipeRecord]   = {}
        self._techs:     dict[str, TechRecord]     = {}

        self._load_all()

    def close(self) -> None:
        """Close the SQLite connection. Safe to call multiple times."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "KnowledgeBase":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Load from DB into memory caches
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._load_entities()
        self._load_resources()
        self._load_fluids()
        self._load_recipes()
        self._load_techs()

    def _load_entities(self) -> None:
        for row in self._conn.execute("SELECT * FROM entities"):
            r = EntityRecord(
                name=row["name"], proto_type=row["proto_type"],
                category=row["category"],
                tile_width=row["tile_width"], tile_height=row["tile_height"],
                has_recipe_slot=bool(row["has_recipe_slot"]),
                ingredient_slots=row["ingredient_slots"],
                output_slots=row["output_slots"],
                is_placeholder=bool(row["is_placeholder"]),
            )
            self._entities[r.name] = r

    def _load_resources(self) -> None:
        for row in self._conn.execute("SELECT * FROM resources"):
            r = ResourceRecord(
                name=row["name"], is_fluid=bool(row["is_fluid"]),
                is_infinite=bool(row["is_infinite"]),
                display_name=row["display_name"],
                is_placeholder=bool(row["is_placeholder"]),
            )
            self._resources[r.name] = r

    def _load_fluids(self) -> None:
        for row in self._conn.execute("SELECT * FROM fluids"):
            r = FluidRecord(
                name=row["name"], temperature=row["temperature"],
                default_temperature=row["default_temperature"],
                max_temperature=row["max_temperature"],
                is_fuel=bool(row["is_fuel"]),
                fuel_value_mj=row["fuel_value_mj"],
                emissions_multiplier=row["emissions_multiplier"],
                is_placeholder=bool(row["is_placeholder"]),
            )
            self._fluids[r.registry_key] = r

    def _load_recipes(self) -> None:
        rows = {row["name"]: row for row in self._conn.execute("SELECT * FROM recipes")}
        if not rows:
            return
        ingredients: dict[str, list[IngredientRecord]] = {n: [] for n in rows}
        for row in self._conn.execute(
            "SELECT * FROM recipe_ingredients ORDER BY recipe_name, position"
        ):
            if row["recipe_name"] in ingredients:
                ingredients[row["recipe_name"]].append(IngredientRecord(
                    name=row["item_name"], amount=row["amount"],
                    is_fluid=bool(row["is_fluid"]), temperature=row["temperature"],
                ))
        products: dict[str, list[ProductRecord]] = {n: [] for n in rows}
        for row in self._conn.execute(
            "SELECT * FROM recipe_products ORDER BY recipe_name, position"
        ):
            if row["recipe_name"] in products:
                products[row["recipe_name"]].append(ProductRecord(
                    name=row["item_name"], amount=row["amount"],
                    probability=row["probability"], is_fluid=bool(row["is_fluid"]),
                    temperature=row["temperature"],
                ))
        made_in: dict[str, list[str]] = {n: [] for n in rows}
        for row in self._conn.execute("SELECT * FROM recipe_made_in"):
            if row["recipe_name"] in made_in:
                made_in[row["recipe_name"]].append(row["entity_name"])
        for name, row in rows.items():
            self._recipes[name] = RecipeRecord(
                name=name, category=row["category"],
                crafting_time=row["crafting_time"],
                ingredients=ingredients[name],
                products=products[name],
                made_in=made_in[name],
                enabled_by_default=bool(row["enabled_by_default"]),
                is_placeholder=bool(row["is_placeholder"]),
            )

    def _load_techs(self) -> None:
        rows = {row["name"]: row for row in self._conn.execute("SELECT * FROM techs")}
        if not rows:
            return
        prereqs: dict[str, list[str]] = {n: [] for n in rows}
        for row in self._conn.execute("SELECT * FROM tech_prerequisites"):
            if row["tech_name"] in prereqs:
                prereqs[row["tech_name"]].append(row["prereq"])
        unlocks_r: dict[str, list[str]] = {n: [] for n in rows}
        for row in self._conn.execute("SELECT * FROM tech_unlocks_recipes"):
            if row["tech_name"] in unlocks_r:
                unlocks_r[row["tech_name"]].append(row["recipe_name"])
        unlocks_e: dict[str, list[str]] = {n: [] for n in rows}
        for row in self._conn.execute("SELECT * FROM tech_unlocks_entities"):
            if row["tech_name"] in unlocks_e:
                unlocks_e[row["tech_name"]].append(row["entity_name"])
        for name, row in rows.items():
            self._techs[name] = TechRecord(
                name=name, prerequisites=prereqs[name],
                unlocks_recipes=unlocks_r[name],
                unlocks_entities=unlocks_e[name],
                requires_dlc=bool(row["requires_dlc"]),
                is_placeholder=bool(row["is_placeholder"]),
            )

    # ------------------------------------------------------------------
    # Write to DB
    # ------------------------------------------------------------------

    def _insert_entity(self, r: EntityRecord) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO entities
            (name, proto_type, category, tile_width, tile_height,
             has_recipe_slot, ingredient_slots, output_slots, is_placeholder)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (r.name, r.proto_type, r.category, r.tile_width, r.tile_height,
              int(r.has_recipe_slot), r.ingredient_slots, r.output_slots,
              int(r.is_placeholder)))
        self._conn.commit()

    def _insert_resource(self, r: ResourceRecord) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO resources
            (name, is_fluid, is_infinite, display_name, is_placeholder)
            VALUES (?,?,?,?,?)
        """, (r.name, int(r.is_fluid), int(r.is_infinite),
              r.display_name, int(r.is_placeholder)))
        self._conn.commit()

    def _insert_fluid(self, r: FluidRecord) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO fluids
            (name, temperature, default_temperature, max_temperature,
             is_fuel, fuel_value_mj, emissions_multiplier, is_placeholder)
            VALUES (?,?,?,?,?,?,?,?)
        """, (r.name, r.temperature, r.default_temperature, r.max_temperature,
              int(r.is_fuel), r.fuel_value_mj, r.emissions_multiplier,
              int(r.is_placeholder)))
        self._conn.commit()

    def _insert_recipe(self, r: RecipeRecord) -> None:
        with self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO recipes
                (name, category, crafting_time, enabled_by_default, is_placeholder)
                VALUES (?,?,?,?,?)
            """, (r.name, r.category, r.crafting_time,
                  int(r.enabled_by_default), int(r.is_placeholder)))
            self._conn.execute(
                "DELETE FROM recipe_ingredients WHERE recipe_name=?", (r.name,))
            self._conn.execute(
                "DELETE FROM recipe_products WHERE recipe_name=?", (r.name,))
            self._conn.execute(
                "DELETE FROM recipe_made_in WHERE recipe_name=?", (r.name,))
            self._conn.executemany("""
                INSERT INTO recipe_ingredients
                (recipe_name, item_name, amount, is_fluid, temperature, position)
                VALUES (?,?,?,?,?,?)
            """, [(r.name, i.name, i.amount, int(i.is_fluid), i.temperature, pos)
                  for pos, i in enumerate(r.ingredients)])
            self._conn.executemany("""
                INSERT INTO recipe_products
                (recipe_name, item_name, amount, probability, is_fluid, temperature, position)
                VALUES (?,?,?,?,?,?,?)
            """, [(r.name, p.name, p.amount, p.probability,
                   int(p.is_fluid), p.temperature, pos)
                  for pos, p in enumerate(r.products)])
            self._conn.executemany("""
                INSERT OR IGNORE INTO recipe_made_in (recipe_name, entity_name)
                VALUES (?,?)
            """, [(r.name, e) for e in r.made_in])

    def _insert_tech(self, r: TechRecord) -> None:
        with self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO techs (name, requires_dlc, is_placeholder)
                VALUES (?,?,?)
            """, (r.name, int(r.requires_dlc), int(r.is_placeholder)))
            self._conn.execute(
                "DELETE FROM tech_prerequisites WHERE tech_name=?", (r.name,))
            self._conn.execute(
                "DELETE FROM tech_unlocks_recipes WHERE tech_name=?", (r.name,))
            self._conn.execute(
                "DELETE FROM tech_unlocks_entities WHERE tech_name=?", (r.name,))
            self._conn.executemany("""
                INSERT OR IGNORE INTO tech_prerequisites (tech_name, prereq)
                VALUES (?,?)
            """, [(r.name, p) for p in r.prerequisites])
            self._conn.executemany("""
                INSERT OR IGNORE INTO tech_unlocks_recipes (tech_name, recipe_name)
                VALUES (?,?)
            """, [(r.name, rec) for rec in r.unlocks_recipes])
            self._conn.executemany("""
                INSERT OR IGNORE INTO tech_unlocks_entities (tech_name, entity_name)
                VALUES (?,?)
            """, [(r.name, e) for e in r.unlocks_entities])

    # ------------------------------------------------------------------
    # Lua query helper
    # ------------------------------------------------------------------

    def _query(self, lua_expr: str) -> Optional[dict]:
        if self._query_fn is None:
            return None
        try:
            raw = self._query_fn(lua_expr)
            if not raw:
                return None
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("ok") is False:
                logger.debug("Lua query failed: %s", data.get("reason"))
                return None
            return data
        except Exception as exc:
            logger.debug("Lua query error for %r: %s", lua_expr, exc)
            return None

    # ------------------------------------------------------------------
    # Entity registry
    # ------------------------------------------------------------------

    def ensure_entity(self, name: str) -> EntityRecord:
        existing = self._entities.get(name)
        if existing is not None and not (existing.is_placeholder and self._query_fn):
            return existing
        record = self._discover_entity(name)
        self._entities[name] = record
        self._insert_entity(record)
        return record

    def _discover_entity(self, name: str) -> EntityRecord:
        data = self._query(f'rcon.print(fa.get_entity_prototype("{name}"))')
        if data is None:
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
        existing = self._resources.get(name)
        if existing is not None and not (existing.is_placeholder and self._query_fn):
            return existing
        record = self._discover_resource(name)
        self._resources[name] = record
        self._insert_resource(record)
        return record

    def _discover_resource(self, name: str) -> ResourceRecord:
        data = self._query(f'rcon.print(fa.get_resource_prototype("{name}"))')
        if data is None:
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
        key = f"{name}@{temperature}" if temperature is not None else name
        existing = self._fluids.get(key)
        if existing is not None and not (existing.is_placeholder and self._query_fn):
            return existing
        record = self._discover_fluid(name, temperature)
        self._fluids[key] = record
        self._insert_fluid(record)
        return record

    def _discover_fluid(self, name: str,
                        temperature: Optional[int]) -> FluidRecord:
        data = self._query(f'rcon.print(fa.get_fluid_prototype("{name}"))')
        if data is None:
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
        existing = self._recipes.get(name)
        if existing is not None and not (existing.is_placeholder and self._query_fn):
            return existing
        record = self._discover_recipe(name)
        self._recipes[name] = record
        self._insert_recipe(record)
        return record

    def _discover_recipe(self, name: str) -> RecipeRecord:
        data = self._query(f'rcon.print(fa.get_recipe_prototype("{name}"))')
        if data is None:
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
        rows = self._conn.execute(
            "SELECT DISTINCT recipe_name FROM recipe_products WHERE item_name=?",
            (product_name,)
        ).fetchall()
        return [self._recipes[r["recipe_name"]] for r in rows
                if r["recipe_name"] in self._recipes]

    def recipes_for_ingredient(self, ingredient_name: str) -> list[RecipeRecord]:
        """All known recipes that consume the given item or fluid."""
        rows = self._conn.execute(
            "SELECT DISTINCT recipe_name FROM recipe_ingredients WHERE item_name=?",
            (ingredient_name,)
        ).fetchall()
        return [self._recipes[r["recipe_name"]] for r in rows
                if r["recipe_name"] in self._recipes]

    def recipes_made_in(self, entity_name: str) -> list[RecipeRecord]:
        """All known recipes that can be crafted in the given entity."""
        rows = self._conn.execute(
            "SELECT recipe_name FROM recipe_made_in WHERE entity_name=?",
            (entity_name,)
        ).fetchall()
        return [self._recipes[r["recipe_name"]] for r in rows
                if r["recipe_name"] in self._recipes]

    # ------------------------------------------------------------------
    # Tech registry
    # ------------------------------------------------------------------

    def ensure_tech(self, name: str) -> TechRecord:
        existing = self._techs.get(name)
        if existing is not None and not (existing.is_placeholder and self._query_fn):
            return existing
        record = self._discover_tech(name)
        self._techs[name] = record
        self._insert_tech(record)
        return record

    def _discover_tech(self, name: str) -> TechRecord:
        data = self._query(f'rcon.print(fa.get_technology("{name}"))')
        if data is None:
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
        return list(record.prerequisites) if record and not record.is_placeholder else []

    def all_prerequisites(self, name: str) -> set[str]:
        result: set[str] = set()
        stack = self.prerequisites(name)
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.add(current)
            stack.extend(self.prerequisites(current))
        return result

    def techs_unlocking_recipe(self, recipe_name: str) -> list[TechRecord]:
        """All known techs that unlock the given recipe."""
        rows = self._conn.execute(
            "SELECT tech_name FROM tech_unlocks_recipes WHERE recipe_name=?",
            (recipe_name,)
        ).fetchall()
        return [self._techs[r["tech_name"]] for r in rows
                if r["tech_name"] in self._techs]

    # ------------------------------------------------------------------
    # Cross-domain queries
    # ------------------------------------------------------------------

    def production_chain(self, target_item: str) -> set[str]:
        """
        Full recursive ingredient closure for target_item.

        Returns the set of all item/fluid names needed (transitively) to produce
        target_item from known recipes. Does not include target_item itself.
        Returns an empty set if no recipe for target_item is known.
        """
        rows = self._conn.execute("""
            WITH RECURSIVE chain(item) AS (
                SELECT ?
                UNION
                SELECT ri.item_name
                FROM recipe_ingredients ri
                JOIN recipe_products rp ON rp.recipe_name = ri.recipe_name
                JOIN chain c ON c.item = rp.item_name
            )
            SELECT item FROM chain WHERE item != ?
        """, (target_item, target_item)).fetchall()
        return {r["item"] for r in rows}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------


    def enrich_placeholders(self) -> dict[str, int]:
        """
        Re-query Factorio for every placeholder record across all five registries.

        Called once a live query_fn becomes available (e.g. after RCON connects).
        Safe to call at any time — non-placeholder records are untouched.

        Returns a dict of counts showing how many placeholders were resolved
        per domain, for logging.
        """
        if self._query_fn is None:
            return {"entities": 0, "resources": 0, "fluids": 0,
                    "recipes": 0, "techs": 0}

        counts = {"entities": 0, "resources": 0, "fluids": 0,
                  "recipes": 0, "techs": 0}

        for name in list(self._entities):
            if self._entities[name].is_placeholder:
                self.ensure_entity(name)
                if not self._entities[name].is_placeholder:
                    counts["entities"] += 1

        for name in list(self._resources):
            if self._resources[name].is_placeholder:
                self.ensure_resource(name)
                if not self._resources[name].is_placeholder:
                    counts["resources"] += 1

        for key, record in list(self._fluids.items()):
            if record.is_placeholder:
                self.ensure_fluid(record.name, record.temperature)
                if not self._fluids[key].is_placeholder:
                    counts["fluids"] += 1

        for name in list(self._recipes):
            if self._recipes[name].is_placeholder:
                self.ensure_recipe(name)
                if not self._recipes[name].is_placeholder:
                    counts["recipes"] += 1

        for name in list(self._techs):
            if self._techs[name].is_placeholder:
                self.ensure_tech(name)
                if not self._techs[name].is_placeholder:
                    counts["techs"] += 1

        return counts

    def summary(self) -> dict:
        return {
            "entities":  len(self._entities),
            "resources": len(self._resources),
            "fluids":    len(self._fluids),
            "recipes":   len(self._recipes),
            "techs":     len(self._techs),
            "data_dir":  str(self._dir),
            "db_path":   str(self._dir / _DB_NAME),
        }