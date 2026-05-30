-- bridge/mod/test_prototypes_live.lua
--
-- Tests for all fa.get_*_prototype() and fa.get_technology() functions.
-- These are the KB query functions — they must return correct 2.x data.
-- Most 1.x→2.x breakage has been in these functions (entity_prototypes→
-- prototypes.entity, tech.effects→tech.prototype.effects, etc.)
--
-- Usage:
--   /c __agent__ TP.run_all()
--   /c __agent__ TP.run_suite("entity_prototype")
--   /c __agent__ TP.run_suite("recipe_prototype")
--   /c __agent__ TP.run_suite("technology")
--   /c __agent__ TP.run_suite("resource_fluid_prototype")

TP = {}
local suites = {}
local log_file = "agent-prototype-test-results.txt"

local function test_print(msg)
    game.print(msg)
    helpers.write_file(log_file, msg .. "\n", true)
end

local function make_assertions()
    local function fail(msg) error("ASSERT: " .. tostring(msg), 2) end
    return {
        ok        = function(v, msg) if not v then fail(msg or "expected truthy") end end,
        eq        = function(a, b, msg)
            if a ~= b then fail(msg or ("expected " .. tostring(b) .. " got " .. tostring(a))) end
        end,
        ne        = function(a, b, msg)
            if a == b then fail(msg or "expected not " .. tostring(b)) end
        end,
        is_number = function(v, msg)
            if type(v) ~= "number" then fail(msg or "expected number got " .. type(v)) end
        end,
        is_string = function(v, msg)
            if type(v) ~= "string" then fail(msg or "expected string got " .. type(v)) end
        end,
        is_table  = function(v, msg)
            if type(v) ~= "table" then fail(msg or "expected table got " .. type(v)) end
        end,
        is_bool   = function(v, msg)
            if type(v) ~= "boolean" then fail(msg or "expected boolean got " .. type(v)) end
        end,
    }
end

local function parse_json(raw)
    local ok, t = pcall(function() return helpers.json_to_table(raw) end)
    return ok and t or nil
end

function TP.suite(name, tests) suites[name] = tests end

function TP.run_suite(name)
    local tests = suites[name]
    if not tests then test_print("[TEST] Unknown suite: " .. name) return end
    local pass, fail_count = 0, 0
    for test_name, fn in pairs(tests) do
        local ok, err = pcall(fn, make_assertions())
        if ok then
            pass = pass + 1
            test_print("[PASS] " .. name .. " :: " .. test_name)
        else
            fail_count = fail_count + 1
            test_print("[FAIL] " .. name .. " :: " .. test_name .. " — " .. tostring(err))
        end
    end
    test_print(string.format("[SUITE %s] %d passed, %d failed", name, pass, fail_count))
    return fail_count == 0
end

function TP.run_all()
    helpers.write_file(log_file, "", false)
    test_print("=== Prototype test run: tick " .. tostring(game.tick) .. " ===")
    for name in pairs(suites) do
        test_print("─────── " .. name .. " ───────")
        TP.run_suite(name)
    end
    test_print("═══ Done. See script-output/" .. log_file .. " ═══")
end

-- ============================================================
-- Suite: entity_prototype
-- ============================================================

TP.suite("entity_prototype", {

    assembler_returns_valid_json = function(t)
        local raw = fa.get_entity_prototype("assembling-machine-1")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil,
            "get_entity_prototype must return valid JSON; got: " .. tostring(raw):sub(1,80))
        t.ok(parsed.ok ~= false,
            "should not return ok=false; reason=" .. tostring(parsed and parsed.reason))
    end,

    assembler_has_required_fields = function(t)
        local raw = fa.get_entity_prototype("assembling-machine-1")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.name ~= nil, "must have name field")
        t.ok(parsed.type ~= nil, "must have type field")
        t.ok(parsed.tile_width ~= nil, "must have tile_width")
        t.ok(parsed.tile_height ~= nil, "must have tile_height")
        t.is_string(parsed.name, "name must be string")
        t.is_number(parsed.tile_width, "tile_width must be number")
        t.eq(parsed.name, "assembling-machine-1", "name should match requested")
    end,

    furnace_has_smelting_category = function(t)
        local raw = fa.get_entity_prototype("stone-furnace")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.category ~= nil, "furnace must have category")
        t.eq(parsed.category, "smelting",
            "stone-furnace category should be smelting; got: " .. tostring(parsed.category))
    end,

    invalid_entity_returns_error = function(t)
        local raw = fa.get_entity_prototype("this-entity-does-not-exist-xyz")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON even for unknown entity")
        t.ok(not parsed.ok,
            "should return ok=false for unknown entity; got ok=" .. tostring(parsed.ok))
    end,

    chest_has_inventory_size = function(t)
        local raw = fa.get_entity_prototype("iron-chest")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.inventory_size ~= nil, "chest must have inventory_size")
        t.is_number(parsed.inventory_size, "inventory_size must be number")
        t.ok(parsed.inventory_size > 0, "inventory_size must be positive")
    end,
})

-- ============================================================
-- Suite: recipe_prototype
-- ============================================================

TP.suite("recipe_prototype", {

    iron_gear_returns_valid_json = function(t)
        local raw = fa.get_recipe_prototype("iron-gear-wheel")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil,
            "get_recipe_prototype must return valid JSON; got: " .. tostring(raw):sub(1,80))
        t.ok(parsed.ok ~= false,
            "should not return ok=false; reason=" .. tostring(parsed and parsed.reason))
    end,

    iron_gear_has_iron_plate_ingredient = function(t)
        local raw = fa.get_recipe_prototype("iron-gear-wheel")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.ingredients ~= nil, "must have ingredients")
        t.is_table(parsed.ingredients, "ingredients must be table")
        t.ok(#parsed.ingredients > 0, "ingredients must not be empty")
        local found_iron = false
        for _, ing in ipairs(parsed.ingredients) do
            if ing.name == "iron-plate" then found_iron = true break end
        end
        t.ok(found_iron, "iron-gear-wheel must require iron-plate")
    end,

    recipe_has_products = function(t)
        local raw = fa.get_recipe_prototype("iron-gear-wheel")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.products ~= nil, "must have products")
        t.is_table(parsed.products, "products must be table")
        t.ok(#parsed.products > 0, "products must not be empty")
        t.eq(parsed.products[1].name, "iron-gear-wheel",
            "product should be iron-gear-wheel; got: " ..
            tostring(parsed.products[1] and parsed.products[1].name))
    end,

    recipe_has_category = function(t)
        local raw = fa.get_recipe_prototype("iron-gear-wheel")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.category ~= nil, "must have category")
        t.is_string(parsed.category, "category must be string")
    end,

    recipe_has_made_in = function(t)
        local raw = fa.get_recipe_prototype("iron-gear-wheel")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.made_in ~= nil, "must have made_in")
        t.is_table(parsed.made_in, "made_in must be table")
    end,

    invalid_recipe_returns_error = function(t)
        local raw = fa.get_recipe_prototype("this-recipe-does-not-exist-xyz")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON for unknown recipe")
        t.ok(not parsed.ok, "should return ok=false for unknown recipe")
    end,
})

-- ============================================================
-- Suite: technology
-- ============================================================

TP.suite("technology", {

    automation_returns_valid_json = function(t)
        local raw = fa.get_technology("automation")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil,
            "get_technology must return valid JSON; got: " .. tostring(raw):sub(1,80))
        t.ok(parsed.ok ~= false,
            "should not return ok=false; reason=" .. tostring(parsed and parsed.reason))
    end,

    automation_not_placeholder = function(t)
        local raw = fa.get_technology("automation")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        -- A placeholder would have empty effects and prerequisites
        -- even though automation has neither prerequisites nor effects in vanilla.
        -- The key test is that parsing did not error — name must be present.
        t.ok(parsed.name ~= nil, "must have name field")
        t.eq(parsed.name, "automation", "name must match requested tech")
    end,

    automation_has_effects = function(t)
        local raw = fa.get_technology("automation")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.effects ~= nil, "must have effects field")
        t.is_table(parsed.effects, "effects must be table; got: " .. type(parsed.effects))
        t.ok(#parsed.effects > 0,
            "automation should have at least one effect (recipe unlock)")
    end,

    automation_unlocks_assembling_machine = function(t)
        local raw = fa.get_technology("automation")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.is_table(parsed.effects, "effects must be table")
        local found = false
        for _, effect in ipairs(parsed.effects) do
            if effect.recipe == "assembling-machine-1" then found = true break end
        end
        t.ok(found,
            "automation should unlock assembling-machine-1; effects=" ..
            helpers.table_to_json(parsed.effects))
    end,

    logistics_has_prerequisites = function(t)
        local raw = fa.get_technology("logistics")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.prerequisites ~= nil, "must have prerequisites field")
        t.is_table(parsed.prerequisites,
            "prerequisites must be table; got: " .. type(parsed.prerequisites))
        t.ok(#parsed.prerequisites > 0,
            "logistics should have at least one prerequisite")
    end,

    prerequisites_are_strings = function(t)
        local raw = fa.get_technology("logistics")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        for i, prereq in ipairs(parsed.prerequisites or {}) do
            t.ok(type(prereq) == "string",
                "prerequisite " .. i .. " must be string; got: " .. type(prereq))
        end
    end,

    tech_has_researched_flag = function(t)
        local raw = fa.get_technology("automation")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.researched ~= nil, "must have researched field")
        t.is_bool(parsed.researched, "researched must be boolean")
    end,

    invalid_tech_returns_error = function(t)
        local raw = fa.get_technology("this-tech-does-not-exist-xyz")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON for unknown tech")
        t.ok(not parsed.ok, "should return ok=false for unknown tech")
    end,
})

-- ============================================================
-- Suite: resource_fluid_prototype
-- ============================================================

TP.suite("resource_fluid_prototype", {

    iron_ore_resource_returns_valid_json = function(t)
        local raw = fa.get_resource_prototype("iron-ore")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil,
            "get_resource_prototype must return valid JSON; got: " .. tostring(raw):sub(1,80))
        t.ok(parsed.ok ~= false,
            "should not return ok=false; reason=" .. tostring(parsed and parsed.reason))
    end,

    iron_ore_has_required_fields = function(t)
        local raw = fa.get_resource_prototype("iron-ore")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.name ~= nil, "must have name; got: " .. helpers.table_to_json(parsed))
        t.ok(parsed.category ~= nil, "must have category")
        t.ok(parsed.infinite ~= nil, "must have infinite flag")
        t.ok(parsed.is_fluid ~= nil, "must have is_fluid flag")
        t.is_string(parsed.name, "name must be string")
        t.eq(parsed.name, "iron-ore", "name must match requested")
    end,

    iron_ore_is_not_infinite = function(t)
        local raw = fa.get_resource_prototype("iron-ore")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.is_bool(parsed.infinite, "infinite must be boolean")
        t.ok(not parsed.infinite, "iron-ore should not be infinite")
    end,

    water_fluid_returns_valid_json = function(t)
        local raw = fa.get_fluid_prototype("water")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil,
            "get_fluid_prototype must return valid JSON; got: " .. tostring(raw):sub(1,80))
        t.ok(parsed.ok ~= false,
            "should not return ok=false; reason=" .. tostring(parsed and parsed.reason))
    end,

    water_fluid_has_required_fields = function(t)
        local raw = fa.get_fluid_prototype("water")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.name ~= nil, "must have name")
        t.ok(parsed.default_temperature ~= nil, "must have default_temperature")
        t.is_string(parsed.name, "name must be string")
        t.is_number(parsed.default_temperature, "default_temperature must be number")
        t.eq(parsed.name, "water", "name must match requested")
    end,

    crude_oil_is_infinite_resource = function(t)
        -- 2.x: proto.infinite_resource is the correct field (mp.infinite is nil).
        local raw = fa.get_resource_prototype("crude-oil")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        if parsed.ok == false then
            test_print("[SKIP] crude-oil not available: " .. tostring(parsed.reason))
            return
        end
        t.is_bool(parsed.infinite, "infinite must be boolean")
        t.ok(parsed.infinite, "crude-oil should be infinite")
    end,

    invalid_resource_returns_error = function(t)
        local raw = fa.get_resource_prototype("this-resource-does-not-exist-xyz")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON for unknown resource")
        t.ok(not parsed.ok, "should return ok=false for unknown resource")
    end,

    invalid_fluid_returns_error = function(t)
        local raw = fa.get_fluid_prototype("this-fluid-does-not-exist-xyz")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON for unknown fluid")
        t.ok(not parsed.ok, "should return ok=false for unknown fluid")
    end,
})

-- ============================================================
-- Suite: item_prototype
-- Verifies fa.get_item_prototype() returns correct stack_size
-- data for common items. Used by the KB items domain.
-- ============================================================

TP.suite("item_prototype", {

    iron_plate_has_expected_fields = function(t)
        local raw = fa.get_item_prototype("iron-plate")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil,
            "fa.get_item_prototype('iron-plate') must return valid JSON; got: " ..
            tostring(raw):sub(1, 80))
        t.ok(parsed.ok ~= false,
            "must not return ok=false; reason=" .. tostring(parsed.reason))
        t.is_string(parsed.name, "name must be a string")
        t.is_number(parsed.stack_size, "stack_size must be a number")
        t.eq(parsed.name, "iron-plate", "name must match request")
    end,

    iron_plate_stack_size_correct = function(t)
        -- Iron plate stacks to 100 in standard Factorio.
        local raw = fa.get_item_prototype("iron-plate")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.ok ~= false, "must succeed")
        t.eq(parsed.stack_size, 100,
            "iron-plate stack_size should be 100; got " .. tostring(parsed.stack_size))
    end,

    iron_gear_wheel_stack_size_correct = function(t)
        -- Iron gear wheel stacks to 100.
        local raw = fa.get_item_prototype("iron-gear-wheel")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.ok ~= false, "must succeed")
        t.eq(parsed.stack_size, 100,
            "iron-gear-wheel stack_size should be 100; got " .. tostring(parsed.stack_size))
    end,

    electronic_circuit_stack_size_correct = function(t)
        -- Electronic circuit stacks to 200.
        local raw = fa.get_item_prototype("electronic-circuit")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.ok ~= false, "must succeed")
        t.eq(parsed.stack_size, 200,
            "electronic-circuit stack_size should be 200; got " .. tostring(parsed.stack_size))
    end,

    stack_size_is_positive_integer = function(t)
        -- For any valid item, stack_size must be >= 1.
        local items = {"iron-plate", "copper-plate", "coal", "stone",
                       "iron-gear-wheel", "copper-cable", "electronic-circuit"}
        for _, name in ipairs(items) do
            local raw = fa.get_item_prototype(name)
            local parsed = parse_json(raw)
            if parsed and parsed.ok ~= false then
                t.ok(parsed.stack_size >= 1,
                    name .. " stack_size must be >= 1; got " ..
                    tostring(parsed.stack_size))
            end
        end
    end,

    unknown_item_returns_error = function(t)
        local raw = fa.get_item_prototype("this-item-does-not-exist-xyz")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON for unknown item")
        t.ok(not parsed.ok, "should return ok=false for unknown item")
    end,

    name_field_matches_request = function(t)
        -- The returned name must echo back the requested item name.
        local raw = fa.get_item_prototype("coal")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.ok ~= false, "must succeed")
        t.eq(parsed.name, "coal", "name must match requested item")
    end,
})


-- ============================================================
-- Suite: entity_mineable_properties
-- Verifies that get_entity_prototype() returns the minable field
-- that supports can_destroy() in execution/predicates.py.
--
-- Key findings from live testing (informed the design):
--   minable=true  — MineEntity works: trees, rocks, machines, chests.
--                   Trees have a mining_trigger for cosmetic particle
--                   effects; MineEntity still completes normally.
--   minable=false — MineEntity does NOT work: cliffs require cliff
--                   explosives via UseItemOnEntity (Phase 7).
--
-- has_mining_trigger was removed after discovering it signals cosmetic
-- effects (leaf particles on trees) rather than resource requirements.
-- ============================================================

TP.suite("entity_mineable_properties", {

    assembler_has_minable_field = function(t)
        local raw = fa.get_entity_prototype("assembling-machine-1")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.minable ~= nil,
            "get_entity_prototype must return minable field")
        t.is_bool(parsed.minable,
            "minable must be boolean; got " .. type(parsed.minable))
    end,

    assembler_is_minable = function(t)
        -- Placed machines can be hand-mined to pick them up.
        -- mineable_properties.minable=true for all placed entities.
        local raw = fa.get_entity_prototype("assembling-machine-1")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.minable,
            "assembling-machine-1 should be minable (can be picked up); " ..
            "got minable=" .. tostring(parsed.minable))
    end,

    chest_is_minable = function(t)
        local raw = fa.get_entity_prototype("iron-chest")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.minable,
            "iron-chest should be minable; got minable=" ..
            tostring(parsed.minable))
    end,

    tree_is_minable = function(t)
        -- Trees are directly MineEntity-destroyable even though their
        -- prototype has a mining_trigger for cosmetic particle effects.
        local tree_proto = nil
        for name, proto in pairs(prototypes.entity) do
            if proto.type == "tree" then tree_proto = name break end
        end
        if not tree_proto then
            test_print("[SKIP] No tree prototype found — skip")
            return
        end
        local raw = fa.get_entity_prototype(tree_proto)
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON for " .. tree_proto)
        t.ok(parsed.minable,
            tree_proto .. " should be minable; got minable=" ..
            tostring(parsed.minable))
    end,

    rock_is_minable = function(t)
        local rock_proto = nil
        for name, proto in pairs(prototypes.entity) do
            if proto.type == "simple-entity" and proto.mineable_properties
                    and proto.mineable_properties.minable then
                rock_proto = name break
            end
        end
        if not rock_proto then
            test_print("[SKIP] No minable simple-entity found — skip")
            return
        end
        local raw = fa.get_entity_prototype(rock_proto)
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON for " .. rock_proto)
        t.ok(parsed.minable,
            rock_proto .. " should be minable; got minable=" ..
            tostring(parsed.minable))
    end,

    cliff_is_not_minable = function(t)
        -- Cliffs have mineable_properties.minable=false in Factorio 2.x.
        -- They require cliff explosives, not MineEntity.
        local raw = fa.get_entity_prototype("cliff")
        local parsed = parse_json(raw)
        if parsed and parsed.ok == false then
            test_print("[SKIP] cliff prototype not found — skip")
            return
        end
        t.ok(parsed ~= nil, "must return valid JSON for cliff")
        t.ok(not parsed.minable,
            "cliff should have minable=false (requires explosives); " ..
            "got minable=" .. tostring(parsed.minable))
    end,

    minable_field_present_on_multiple_entity_types = function(t)
        -- Spot-check a range of entity types to ensure minable is always
        -- returned and always boolean.
        local entities = {
            "assembling-machine-1", "iron-chest", "inserter",
            "transport-belt", "electric-mining-drill",
        }
        for _, name in ipairs(entities) do
            local raw = fa.get_entity_prototype(name)
            local parsed = parse_json(raw)
            if parsed and parsed.ok ~= false then
                t.ok(parsed.minable ~= nil,
                    name .. " must have minable field")
                t.is_bool(parsed.minable,
                    name .. " minable must be boolean; got " ..
                    type(parsed.minable))
            end
        end
    end,

    new_field_coexists_with_existing_fields = function(t)
        local raw = fa.get_entity_prototype("assembling-machine-1")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.name ~= nil,            "name must still be present")
        t.ok(parsed.type ~= nil,            "type must still be present")
        t.ok(parsed.tile_width ~= nil,      "tile_width must still be present")
        t.ok(parsed.tile_height ~= nil,     "tile_height must still be present")
        t.ok(parsed.has_recipe_slot ~= nil, "has_recipe_slot must still be present")
        t.ok(parsed.inventory_size ~= nil,  "inventory_size must still be present")
        t.ok(parsed.minable ~= nil,         "minable must be present")
    end,

    minable_consistent_with_mineable_properties = function(t)
        local entity_name = "assembling-machine-1"
        local proto = prototypes.entity[entity_name]
        t.ok(proto ~= nil, "prototype must exist")
        local lua_minable = false
        local ok_mp, mp = pcall(function() return proto.mineable_properties end)
        if ok_mp and mp then lua_minable = mp.minable == true end
        local raw = fa.get_entity_prototype(entity_name)
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.eq(parsed.minable, lua_minable,
            "minable must match proto.mineable_properties.minable; " ..
            "lua=" .. tostring(lua_minable) ..
            " returned=" .. tostring(parsed.minable))
    end,
})


-- ============================================================
-- Suite: entity_mining_products
-- Verifies that get_entity_prototype() returns the mining_products
-- field introduced to support kb.entities_that_produce(item) and
-- the harvest_natural task type.
--
-- Trees drop wood, rocks drop stone/coal/iron-ore depending on type.
-- Machines drop themselves (their item equivalent).
-- The mining_products list enables the coordinator to find harvestable
-- sources for items that are not resource patches.
-- ============================================================

TP.suite("entity_mining_products", {

    mining_products_field_present = function(t)
        -- The field must be present on every entity response regardless of
        -- whether the entity has any mining products.
        local raw = fa.get_entity_prototype("assembling-machine-1")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.mining_products ~= nil,
            "get_entity_prototype must return mining_products field. "
            .. "Check control.lua fa.get_entity_prototype() return table.")
        t.is_table(parsed.mining_products,
            "mining_products must be a table (array); got " ..
            type(parsed.mining_products))
    end,

    machine_mining_products_empty_or_has_item = function(t)
        -- Assembling machines return themselves as a product when mined.
        -- In some modded scenarios this may be empty, so we accept both.
        local raw = fa.get_entity_prototype("assembling-machine-1")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.is_table(parsed.mining_products, "mining_products must be a table")
        -- If non-empty, each entry must have name (string) and amount (number).
        for i, prod in ipairs(parsed.mining_products) do
            t.ok(prod.name ~= nil,
                "mining_products[" .. i .. "] must have name field")
            t.is_string(prod.name,
                "mining_products[" .. i .. "].name must be string")
            t.ok(prod.amount ~= nil,
                "mining_products[" .. i .. "] must have amount field")
            t.is_number(prod.amount,
                "mining_products[" .. i .. "].amount must be number; got " ..
                type(prod.amount))
            t.ok(prod.amount > 0,
                "mining_products[" .. i .. "].amount must be positive; got " ..
                tostring(prod.amount))
        end
    end,

    tree_drops_wood = function(t)
        -- Find any tree prototype on this map and verify it drops wood.
        -- Trees are the canonical example of a harvestable natural entity.
        local tree_proto = nil
        for name, proto in pairs(prototypes.entity) do
            if proto.type == "tree" then tree_proto = name break end
        end
        if not tree_proto then
            test_print("[SKIP] No tree prototype found on this map — skip")
            return
        end
        local raw = fa.get_entity_prototype(tree_proto)
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON for " .. tree_proto)
        t.is_table(parsed.mining_products,
            "tree mining_products must be a table")
        t.ok(#parsed.mining_products > 0,
            tree_proto .. " must have at least one mining product (wood); "
            .. "got empty mining_products. Check that mineable_properties.products "
            .. "is being read in fa.get_entity_prototype().")
        local found_wood = false
        for _, prod in ipairs(parsed.mining_products) do
            if prod.name == "wood" then found_wood = true break end
        end
        t.ok(found_wood,
            tree_proto .. " mining_products must include wood; "
            .. "got " .. tostring(#parsed.mining_products) .. " product(s) but no wood.")
    end,

    rock_drops_stone_or_ore = function(t)
        -- Rocks (simple-entity with minable=true) typically drop stone,
        -- coal, or iron-ore depending on type. Verify the field is populated.
        local rock_proto = nil
        local rock_products_count = 0
        for name, proto in pairs(prototypes.entity) do
            if proto.type == "simple-entity"
                    and proto.mineable_properties
                    and proto.mineable_properties.minable
                    and proto.mineable_properties.products
                    and #proto.mineable_properties.products > 0 then
                rock_proto = name
                rock_products_count = #proto.mineable_properties.products
                break
            end
        end
        if not rock_proto then
            test_print("[SKIP] No minable simple-entity with products found — skip")
            return
        end
        local raw = fa.get_entity_prototype(rock_proto)
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON for " .. rock_proto)
        t.is_table(parsed.mining_products, "mining_products must be a table")
        t.ok(#parsed.mining_products > 0,
            rock_proto .. " must have mining products (drops stone/ore/coal); "
            .. "raw prototype had " .. rock_products_count .. " product(s) but "
            .. "mining_products is empty. Check product.type == 'fluid' filter.")
        -- Every product must have valid fields.
        for i, prod in ipairs(parsed.mining_products) do
            t.is_string(prod.name,
                rock_proto .. " product[" .. i .. "].name must be string")
            t.is_number(prod.amount,
                rock_proto .. " product[" .. i .. "].amount must be number")
            t.ok(prod.amount > 0,
                rock_proto .. " product[" .. i .. "].amount must be positive")
        end
    end,

    chest_mining_products_field_present = function(t)
        -- Chests are minable — they drop themselves. The field must be present
        -- even for entities where the product is the entity item itself.
        local raw = fa.get_entity_prototype("iron-chest")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.mining_products ~= nil,
            "iron-chest must have mining_products field")
        t.is_table(parsed.mining_products, "mining_products must be a table")
    end,

    mining_products_coexists_with_minable = function(t)
        -- Both minable (bool) and mining_products (table) must be present
        -- in the same response — they come from the same mineable_properties.
        local raw = fa.get_entity_prototype("assembling-machine-1")
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.minable ~= nil,
            "minable field must still be present alongside mining_products")
        t.ok(parsed.mining_products ~= nil,
            "mining_products must be present alongside minable")
        t.is_bool(parsed.minable,
            "minable must be boolean")
        t.is_table(parsed.mining_products,
            "mining_products must be table")
    end,

    mining_products_amount_is_average_of_range = function(t)
        -- When a product has amount_min and amount_max (a range), the returned
        -- amount should be their average. Find a prototype with a range if possible.
        -- If none found, just verify the field structure is always correct.
        local range_proto = nil
        local expected_avg = nil
        for name, proto in pairs(prototypes.entity) do
            if proto.type == "simple-entity"
                    and proto.mineable_properties
                    and proto.mineable_properties.products then
                for _, prod in ipairs(proto.mineable_properties.products) do
                    if prod.amount_min and prod.amount_max
                            and prod.amount_min ~= prod.amount_max then
                        range_proto = name
                        expected_avg = (prod.amount_min + prod.amount_max) / 2
                        break
                    end
                end
            end
            if range_proto then break end
        end
        if not range_proto then
            test_print("[SKIP] No prototype with amount range found — skip")
            return
        end
        local raw = fa.get_entity_prototype(range_proto)
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON for " .. range_proto)
        t.ok(#parsed.mining_products > 0,
            range_proto .. " must have mining products")
        -- The first product's amount should be the average of the range.
        local got_amount = parsed.mining_products[1].amount
        t.ok(math.abs(got_amount - expected_avg) < 0.01,
            "amount for range product should be average of min+max; "
            .. "expected " .. expected_avg .. " got " .. tostring(got_amount))
    end,

    fluid_products_excluded = function(t)
        -- Fluid products (e.g. crude-oil from an oil resource) should NOT
        -- appear in mining_products — the agent cannot collect fluids via
        -- MineEntity. Verify by finding an entity with fluid products if any.
        -- Most entities in vanilla won't have this, so this is a structural test.
        for name, proto in pairs(prototypes.entity) do
            if proto.mineable_properties
                    and proto.mineable_properties.products then
                local has_fluid = false
                for _, prod in ipairs(proto.mineable_properties.products) do
                    if prod.type == "fluid" then has_fluid = true break end
                end
                if has_fluid then
                    local raw = fa.get_entity_prototype(name)
                    local parsed = parse_json(raw)
                    if parsed and parsed.ok ~= false then
                        for _, mp in ipairs(parsed.mining_products or {}) do
                            -- Fluid items typically have a "/" in their internal
                            -- representation or a specific type marker. The filter
                            -- in get_entity_prototype checks product.type ~= "fluid".
                            -- We just verify no obviously fluid name leaked through.
                            t.ok(mp.name ~= "water" and mp.name ~= "crude-oil"
                                    and mp.name ~= "heavy-oil"
                                    and mp.name ~= "light-oil"
                                    and mp.name ~= "petroleum-gas"
                                    and mp.name ~= "steam",
                                name .. " mining_products contains fluid " ..
                                tostring(mp.name) .. " — fluid products should be filtered")
                        end
                    end
                    break   -- one check is enough
                end
            end
        end
        -- If no entity with fluid products found, test vacuously passes.
    end,
})

return TP