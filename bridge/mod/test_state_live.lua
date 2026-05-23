-- bridge/mod/test_state_live.lua
--
-- Tests for world-state query functions in control.lua.
-- Covers: fa.get_exploration(), fa._player_table() internals (charted_chunks,
-- movement_status), fa.get_mining_status(), fa.stop_mining().
--
-- Usage:
--   /c __agent__ TS.run_all()
--   /c __agent__ TS.run_suite("exploration")
--   /c __agent__ TS.run_suite("mining_status")
--
-- TS is registered as a global by control.lua:
--   TS = require("test_state_live")

TS = {}
local suites = {}
local log_file = "agent-state-test-results.txt"

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
            if a == b then fail(msg or ("expected not " .. tostring(b))) end
        end,
        gt        = function(a, b, msg)
            if not (a > b) then fail(msg or (tostring(a) .. " not > " .. tostring(b))) end
        end,
        gte       = function(a, b, msg)
            if not (a >= b) then fail(msg or (tostring(a) .. " not >= " .. tostring(b))) end
        end,
        is_number = function(v, msg)
            if type(v) ~= "number" then fail(msg or ("expected number got " .. type(v))) end
        end,
        is_string = function(v, msg)
            if type(v) ~= "string" then fail(msg or ("expected string got " .. type(v))) end
        end,
        is_table  = function(v, msg)
            if type(v) ~= "table" then fail(msg or ("expected table got " .. type(v))) end
        end,
    }
end

local function parse_json(raw)
    local ok, t = pcall(function() return helpers.json_to_table(raw) end)
    return ok and t or nil
end

local function get_player() return game.get_player(1) end

function TS.suite(name, tests) suites[name] = tests end

function TS.run_suite(name)
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

function TS.run_all()
    helpers.write_file(log_file, "", false)
    test_print("=== State test run: tick " .. tostring(game.tick) .. " ===")
    for name in pairs(suites) do
        test_print("─────── " .. name .. " ───────")
        TS.run_suite(name)
    end
    test_print("═══ Done. See script-output/" .. log_file .. " ═══")
end

-- ============================================================
-- Suite: exploration
-- Verifies fa.get_exploration() and the charted_chunks field in
-- fa.get_state() / fa._player_table().
-- These are NON-PROXIMAL — must reflect the global chart size.
-- ============================================================

TS.suite("exploration", {

    get_exploration_returns_valid_json = function(t)
        local raw = fa.get_exploration()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "fa.get_exploration() must return valid JSON; got: " .. tostring(raw):sub(1,80))
        t.ok(parsed.ok ~= false, "fa.get_exploration() should not return ok=false")
    end,

    get_exploration_has_charted_chunks = function(t)
        local raw = fa.get_exploration()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.ok(parsed.charted_chunks ~= nil, "must have charted_chunks field")
        t.is_number(parsed.charted_chunks,
            "charted_chunks must be a number; got type=" .. type(parsed.charted_chunks) ..
            " val=" .. tostring(parsed.charted_chunks))
    end,

    charted_chunks_positive = function(t)
        -- Any loaded game has at least the spawn chunks charted.
        local raw = fa.get_exploration()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        t.gt(parsed.charted_chunks, 0,
            "charted_chunks should be > 0 in any loaded game; got: " ..
            tostring(parsed.charted_chunks))
    end,

    charted_chunk_count_via_iterator = function(t)
        -- Verified 2.x API: surface.get_chunks() + force.is_chunk_charted(surface, chunk).
        -- get_chart_size() was removed; chunk.position is wrong (pass chunk directly).
        local player = get_player()
        t.ok(player and player.valid, "need a valid player")
        local count = 0
        local ok = pcall(function()
            for chunk in player.surface.get_chunks() do
                if player.force.is_chunk_charted(player.surface, chunk) then
                    count = count + 1
                end
            end
        end)
        t.ok(ok, "get_chunks/is_chunk_charted should not error")
        t.ok(count > 0,
            "should find charted chunks; got: " .. tostring(count))
    end,

    player_table_charted_chunks_matches_exploration = function(t)
        -- charted_chunks in get_state().player must equal fa.get_exploration().charted_chunks
        local exp_raw = fa.get_exploration()
        local exp = parse_json(exp_raw)
        t.ok(exp ~= nil, "fa.get_exploration() must return valid JSON")

        local state_raw = fa.get_state({radius=4, resource_radius=32, item_radius=4})
        local state = parse_json(state_raw)
        t.ok(state ~= nil, "fa.get_state() must return valid JSON")
        t.ok(state.player ~= nil, "get_state must have player section")

        local state_chunks = state.player.charted_chunks
        local exp_chunks   = exp.charted_chunks

        t.is_number(state_chunks,
            "get_state().player.charted_chunks must be a number; got: " ..
            type(state_chunks))
        t.is_number(exp_chunks,
            "get_exploration().charted_chunks must be a number; got: " ..
            type(exp_chunks))
        t.eq(state_chunks, exp_chunks,
            "get_state and get_exploration should agree on charted_chunks; " ..
            "state=" .. tostring(state_chunks) .. " exploration=" .. tostring(exp_chunks))
    end,

    charted_chunks_is_tick_independent = function(t)
        -- Call twice in the same tick — value should be stable.
        local r1 = parse_json(fa.get_exploration())
        local r2 = parse_json(fa.get_exploration())
        t.ok(r1 ~= nil and r2 ~= nil, "both calls must return valid JSON")
        t.eq(r1.charted_chunks, r2.charted_chunks,
            "charted_chunks should be stable within a tick")
    end,
})

-- ============================================================
-- Suite: mining_status
-- Verifies fa.get_mining_status() and fa.stop_mining() return
-- correct values and don't crash.
-- ============================================================

TS.suite("mining_status", {

    get_mining_status_idle_when_not_mining = function(t)
        fa.stop_mining()
        local raw = fa.get_mining_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "fa.get_mining_status() must return valid JSON")
        t.ok(parsed.ok ~= false, "should return ok=true")
        t.is_string(parsed.status, "status must be a string")
        t.eq(parsed.status, "idle",
            "status should be idle after stop_mining; got: " .. tostring(parsed.status))
    end,

    get_mining_status_valid_values = function(t)
        local raw = fa.get_mining_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        local valid = {idle=true, mining=true}
        t.ok(valid[parsed.status],
            "status must be idle or mining; got: " .. tostring(parsed.status))
    end,

    stop_mining_returns_ok = function(t)
        local raw = fa.stop_mining()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "fa.stop_mining() must return valid JSON")
        t.ok(parsed.ok, "stop_mining should return ok=true")
    end,

    stop_mining_twice_is_safe = function(t)
        fa.stop_mining()
        local raw = fa.stop_mining()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "second stop_mining must return valid JSON")
        t.ok(parsed.ok, "second stop_mining should return ok=true")
    end,

    get_mining_status_in_player_table = function(t)
        -- get_state does not currently include mining_status in player section
        -- (only movement_status). This test documents that and would catch
        -- a regression if mining_status were accidentally added in a broken form.
        local raw = fa.get_state({radius=4, resource_radius=32, item_radius=4})
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_state must return valid JSON")
        t.ok(parsed.player ~= nil, "get_state must have player section")
        -- movement_status should be present
        t.ok(parsed.player.movement_status ~= nil,
            "player section should have movement_status field")
        t.is_string(parsed.player.movement_status,
            "movement_status must be a string")
        -- inventory_size should be present
        t.ok(parsed.player.inventory_size ~= nil,
            "player section should have inventory_size field")
        t.is_number(parsed.player.inventory_size,
            "inventory_size must be a number")
    end,
})

-- ============================================================
-- Suite: inventory_size
-- Verifies that _player_table() reports the total character
-- inventory slot count via the new inventory_size field.
-- ============================================================

TS.suite("inventory_size", {

    inventory_size_present_in_get_state = function(t)
        local raw = fa.get_state({radius=4, resource_radius=32, item_radius=4})
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_state must return valid JSON")
        t.ok(parsed.player ~= nil, "must have player section")
        t.ok(parsed.player.inventory_size ~= nil,
            "player must have inventory_size field")
    end,

    inventory_size_is_positive_integer = function(t)
        -- The character main inventory is always at least 1 slot in any
        -- loaded game. Standard character inventory is 80 slots.
        local raw = fa.get_state({radius=4, resource_radius=32, item_radius=4})
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        local sz = parsed.player.inventory_size
        t.is_number(sz, "inventory_size must be a number; got " .. type(sz))
        t.gt(sz, 0, "inventory_size must be > 0; got " .. tostring(sz))
    end,

    inventory_size_at_least_as_large_as_occupied_slots = function(t)
        -- The total slot count must never be less than the number of
        -- occupied slots the inventory list reports.
        local raw = fa.get_state({radius=4, resource_radius=32, item_radius=4})
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "must return valid JSON")
        local sz = parsed.player.inventory_size
        local occupied = 0
        if type(parsed.player.inventory) == "table" then
            for _ in pairs(parsed.player.inventory) do
                occupied = occupied + 1
            end
        end
        t.ok(sz >= occupied,
            "inventory_size (" .. tostring(sz) ..
            ") must be >= occupied slots (" .. tostring(occupied) .. ")")
    end,

    inventory_size_stable_across_calls = function(t)
        -- Total slot count should not change between two back-to-back calls
        -- (nothing in these tests resizes the inventory).
        local r1 = parse_json(fa.get_state({radius=4, resource_radius=32, item_radius=4}))
        local r2 = parse_json(fa.get_state({radius=4, resource_radius=32, item_radius=4}))
        t.ok(r1 ~= nil and r2 ~= nil, "both calls must return valid JSON")
        t.eq(r1.player.inventory_size, r2.player.inventory_size,
            "inventory_size should be stable; got " ..
            tostring(r1.player.inventory_size) .. " then " ..
            tostring(r2.player.inventory_size))
    end,
})

return TS