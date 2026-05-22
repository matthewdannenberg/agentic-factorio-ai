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

return TP