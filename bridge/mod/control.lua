-- bridge/mod/control.lua
--
-- Factorio mod — exposes game state to the Python agent via RCON and accepts
-- action commands. All output is JSON strings produced by helpers.table_to_json().
--
-- Target: Factorio 2.x (Space Age). Not compatible with 1.x.
--
-- 2.x migration notes (changes from 1.x version)
-- ------------------------------------------------
-- • info.json must declare "factorio_version": "2.0"
-- • Logistic chest entity type names renamed:
--     logistic-chest-passive-provider  → passive-provider-chest
--     logistic-chest-active-provider   → active-provider-chest
--     logistic-chest-storage           → storage-chest
--     logistic-chest-requester         → requester-chest
--     logistic-chest-buffer            → buffer-chest
-- • entity.get_recipe() now returns (LuaRecipe?, LuaQualityPrototype?) — two
--   values. We take only the first.
-- • entity.get_health_ratio() removed. Use entity.health / entity.max_health.
-- • player.reach_distance moved to player.character.reach_distance.
-- • force.research_queue direct table assignment removed.
--   Use force.add_research(name) to enqueue; force.cancel_current_research()
--   to clear.
-- • electric_network_statistics.get_flow_count() signature changed:
--     1.x: get_flow_count("output", true)
--     2.x: get_flow_count{name=proto, category="output",
--                         precision_index=defines.flow_precision_index.five_seconds,
--                         count=false}
--   Returns per-tick joules; multiply by 60 for watts, divide by 1000 for kW.
-- • defines.entity_status entries reorganised — guarded with nil-checks.
-- • spawner_data removed from unit-spawner; use unit_count instead.
-- • player.walking_state → player.character.walking_state
-- • move_to now uses all 8 compass directions including diagonals.
--
-- Usage from Python (via RCON /c command)
-- ----------------------------------------
--   /c rcon.print(fa.get_state({radius=32, resource_radius=128, item_radius=16}))
--   /c rcon.print(fa.get_player())
--   /c rcon.print(fa.move_to({x=10, y=20}, true))
--   /c rcon.print(fa.craft_item("iron-plate", 5))

fa = {}

-- ============================================================
-- Destruction event circular buffer
-- ============================================================

local DESTRUCTION_BUFFER_SIZE = 512
local destruction_events = {}

local function push_destruction_event(entry)
    table.insert(destruction_events, entry)
    if #destruction_events > DESTRUCTION_BUFFER_SIZE then
        table.remove(destruction_events, 1)
    end
end

-- ============================================================
-- Event handlers
-- ============================================================

script.on_event(defines.events.on_entity_died, function(event)
    local entity = event.entity
    if not entity or not entity.valid then return end

    local cause = "unknown"
    if event.cause then
        if event.cause.type == "character" then
            cause = "vehicle"
        end
        if event.cause.force and event.cause.force.name == "enemy" then
            cause = "biter"
        end
    end
    if not event.cause then
        local ok, flagged = pcall(function()
            return entity.to_be_deconstructed and entity.to_be_deconstructed()
        end)
        if ok and flagged then
            cause = "deconstruct"
        end
    end

    push_destruction_event({
        name         = entity.name,
        position     = {x = entity.position.x, y = entity.position.y},
        destroyed_at = game.tick,
        cause        = cause,
    })
end)

-- ============================================================
-- Helper utilities
-- ============================================================

local function get_player()
    return game.get_player(1)
end

local function safe_json(t)
    local ok, result = pcall(function() return helpers.table_to_json(t) end)
    if ok then
        return result
    else
        return '{"ok":false,"reason":"json_serialisation_error"}'
    end
end

-- 2.x: defines.entity_status was reorganised. Guard each lookup with a nil
-- sentinel so that removed or renamed entries fall through to "unknown"
-- instead of crashing.
local function entity_status_string(entity)
    if not entity.valid then return "unknown" end
    local s  = entity.status
    local ds = defines.entity_status
    if     s == ds.working                          then return "working"
    elseif s == (ds.normal or -1)                  then return "working"
    elseif s == ds.idle                             then return "idle"
    elseif s == ds.item_ingredient_shortage         then return "item_ingredient_shortage"
    elseif s == ds.fluid_ingredient_shortage        then return "fluid_ingredient_shortage"
    elseif s == ds.no_input_fluid                   then return "no_input_fluid"
    elseif s == ds.no_minable_resources             then return "no_minable_resources"
    elseif s == ds.no_power                         then return "no_power"
    elseif s == ds.not_plugged_in_electric_network  then return "not_plugged_in_electric_network"
    elseif s == ds.no_fuel                          then return "no_fuel"
    elseif s == ds.full_output                      then return "full_output"
    elseif s == (ds.output_full or -1)              then return "full_output"
    else                                            return "unknown"
    end
end

local function inventory_to_list(inventory)
    if not inventory or not inventory.valid then return {} end
    local slots = {}
    for i = 1, #inventory do
        local stack = inventory[i]
        if stack and stack.valid_for_read then
            table.insert(slots, {item = stack.name, count = stack.count})
        end
    end
    return slots
end

local function entity_to_table(entity)
    if not entity or not entity.valid then return nil end
    local t = {
        unit_number = entity.unit_number,
        name        = entity.name,
        position    = {x = entity.position.x, y = entity.position.y},
        direction   = entity.direction or 0,
        status      = entity_status_string(entity),
        energy      = entity.energy or 0.0,
    }
    -- 2.x: get_recipe() returns (LuaRecipe?, LuaQualityPrototype?) — two values.
    -- Capture only the first; pcall guards against entities without recipe support.
    if entity.get_recipe then
        local ok, recipe = pcall(function() return entity.get_recipe() end)
        if ok and recipe then
            t.recipe = recipe.name
        end
    end
    if entity.get_inventory then
        local inv = entity.get_inventory(defines.inventory.assembling_machine_input)
                 or entity.get_inventory(defines.inventory.furnace_source)
                 or entity.get_inventory(defines.inventory.chest)
        if inv then
            t.inventory = inventory_to_list(inv)
        end
    end
    return t
end

-- ============================================================
-- State query functions
-- ============================================================

function fa.get_state(opts)
    opts = opts or {}
    local radius      = opts.radius          or 32
    local res_radius  = opts.resource_radius or 128
    local item_radius = opts.item_radius     or 16

    local player = get_player()
    if not player or not player.valid then
        return '{"ok":false,"reason":"no_player"}'
    end

    local state = {
        tick               = game.tick,
        player             = fa._player_table(player),
        entities           = fa._entities_table(player, radius),
        resource_map       = fa._resource_map_table(player, res_radius),
        ground_items       = fa._ground_items_table(player, item_radius),
        research           = fa._research_table(),
        logistics          = fa._logistics_table(player, radius),
        damaged_entities   = fa._damaged_entities_table(player, radius),
        destroyed_entities = destruction_events,
        threat             = fa._threat_table(),
    }
    destruction_events = {}
    return safe_json(state)
end

function fa.get_player()
    local player = get_player()
    if not player or not player.valid then
        return '{"ok":false,"reason":"no_player"}'
    end
    return safe_json({tick = game.tick, player = fa._player_table(player)})
end

function fa.get_entities(radius)
    radius = radius or 32
    local player = get_player()
    return safe_json({tick = game.tick, entities = fa._entities_table(player, radius)})
end

function fa.get_resource_map(radius)
    radius = radius or 128
    local player = get_player()
    return safe_json({tick = game.tick, resource_map = fa._resource_map_table(player, radius)})
end

function fa.get_ground_items(radius)
    radius = radius or 16
    local player = get_player()
    return safe_json({tick = game.tick, ground_items = fa._ground_items_table(player, radius)})
end

function fa.get_research()
    return safe_json({tick = game.tick, research = fa._research_table()})
end

function fa.get_logistics(radius)
    radius = radius or 32
    local player = get_player()
    return safe_json({tick = game.tick, logistics = fa._logistics_table(player, radius)})
end

function fa.get_damaged_entities(radius)
    radius = radius or 32
    local player = get_player()
    return safe_json({tick = game.tick,
                      damaged_entities = fa._damaged_entities_table(player, radius)})
end

function fa.drain_destruction_events()
    local events = destruction_events
    destruction_events = {}
    return safe_json({tick = game.tick, destroyed_entities = events})
end

function fa.get_threat()
    return safe_json({tick = game.tick, threat = fa._threat_table()})
end

function fa.get_tick()
    return tostring(game.tick)
end

-- Returns current exploration statistics without a full state query.
-- Useful for partial refreshes when only charted area is needed.
-- charted_chunks is NON-PROXIMAL: reflects the force's global chart,
-- not the current scan radius.
function fa.get_exploration()
    local player = get_player()
    if not player or not player.valid then
        return err_response("no_player")
    end
    local charted_chunks = 0
    local ok, val = pcall(function()
        return player.force.get_chart_size(player.surface)
    end)
    if ok and val then charted_chunks = val end
    return safe_json({
        tick           = game.tick,
        charted_chunks = charted_chunks,
    })
end

-- ============================================================
-- Internal section builders
-- ============================================================

function fa._player_table(player)
    local inv = player.get_inventory(defines.inventory.character_main)
    -- charted_chunks: total 32x32 chunks the force has revealed on this surface.
    -- Global to the force -- not scan-radius scoped. Grows monotonically.
    -- Sourced from LuaForce::get_chart_size(surface).
    local charted_chunks = 0
    local ok_cc, cc = pcall(function()
        return player.force.get_chart_size(player.surface)
    end)
    if ok_cc and cc then charted_chunks = cc end
    return {
        position       = {x = player.position.x, y = player.position.y},
        health         = player.character and player.character.health or 100.0,
        inventory      = inventory_to_list(inv),
        reachable      = fa._reachable_ids(player),
        charted_chunks = charted_chunks,
    }
end

function fa._reachable_ids(player)
    local ids   = {}
    -- 2.x: reach_distance is on the character entity, not the player object.
    local reach = (player.character and player.character.reach_distance) or 6
    local entities = player.surface.find_entities_filtered({
        position = player.position,
        radius   = reach,
    })
    for _, e in ipairs(entities) do
        if e.unit_number then
            table.insert(ids, e.unit_number)
        end
    end
    return ids
end

function fa._entities_table(player, radius)
    local entities_list = {}
    local surface = player.surface
    local center  = player.position

    -- Query without a type filter, then keep anything that has a unit_number.
    --
    -- unit_number is only assigned to persistent, player-interactable entities:
    -- assemblers, inserters, poles, chests, modded machines, turrets, trains, etc.
    -- It is NOT assigned to: trees, rocks, cliffs, resource tiles, decoratives,
    -- projectiles, smoke, fish, or other cosmetic/transient entities.
    --
    -- This means the filter is automatic and fully mod-compatible: any building
    -- added by any mod will appear here without code changes, because Factorio
    -- assigns unit_numbers to all such entities by construction.
    --
    -- The one thing this captures that we do not want is the player character
    -- itself (and other characters in multiplayer), which has a unit_number.
    -- We exclude those by type check.
    local found = surface.find_entities_filtered({
        position = center,
        radius   = radius,
    })

    for _, entity in ipairs(found) do
        if entity.unit_number and entity.type ~= "character" then
            local t = entity_to_table(entity)
            if t then table.insert(entities_list, t) end
        end
    end
    return entities_list
end

function fa._resource_map_table(player, radius)
    local patches = {}
    local surface = player.surface
    local found   = surface.find_entities_filtered({
        position = player.position,
        radius   = radius,
        type     = "resource",
    })

    local groups = {}
    for _, res in ipairs(found) do
        local name = res.name
        if not groups[name] then
            groups[name] = {sum_x=0, sum_y=0, count=0, amount=0}
        end
        local g = groups[name]
        g.sum_x  = g.sum_x  + res.position.x
        g.sum_y  = g.sum_y  + res.position.y
        g.count  = g.count  + 1
        g.amount = g.amount + (res.amount or 0)
    end

    for name, g in pairs(groups) do
        table.insert(patches, {
            resource_type = name,
            position      = {x = g.sum_x / g.count, y = g.sum_y / g.count},
            amount        = g.amount,
            size          = g.count,
            observed_at   = game.tick,
        })
    end
    return patches
end

function fa._ground_items_table(player, radius)
    local items = {}
    local found = player.surface.find_entities_filtered({
        position = player.position,
        radius   = radius,
        type     = "item-entity",
    })
    for _, entity in ipairs(found) do
        if entity.stack and entity.stack.valid_for_read then
            table.insert(items, {
                item        = entity.stack.name,
                position    = {x = entity.position.x, y = entity.position.y},
                count       = entity.stack.count,
                observed_at = game.tick,
                age_ticks   = 0,
            })
        end
    end
    return items
end

function fa._research_table()
    local force = game.forces["player"]
    if not force then return {} end

    local unlocked = {}
    for name, tech in pairs(force.technologies) do
        if tech.researched then
            table.insert(unlocked, name)
        end
    end

    local in_progress = nil
    if force.current_research then
        in_progress = force.current_research.name
    end

    -- 2.x: force.research_queue is still a readable sequence of LuaTechnology.
    local queued = {}
    for _, tech in ipairs(force.research_queue or {}) do
        table.insert(queued, tech.name)
    end

    return {
        unlocked           = unlocked,
        in_progress        = in_progress,
        queued             = queued,
        science_per_minute = {},  -- accurate SPM requires a tick accumulator; deferred
    }
end

function fa._logistics_table(player, radius)
    local surface = player.surface
    local center  = player.position

    -- ---- Power grid ----
    -- 2.x: get_flow_count() takes a named-argument table.
    --   category "output" = production (generators), "input" = consumption (machines).
    --   Returns per-tick joules. Multiply by 60 (ticks/s) for watts, ÷1000 for kW.
    local power = {produced_kw=0, consumed_kw=0, accumulated_kj=0, satisfaction=1.0}

    local poles = surface.find_entities_filtered({
        position = center,
        radius   = radius,
        type     = "electric-pole",
    })

    if poles and poles[1] and poles[1].valid then
        local stats = poles[1].electric_network_statistics
        if stats then
            local produced_j_tick = 0
            local consumed_j_tick = 0

            for proto_name, _ in pairs(stats.output_counts) do
                local ok, val = pcall(function()
                    return stats.get_flow_count{
                        name            = proto_name,
                        category        = "output",
                        precision_index = defines.flow_precision_index.five_seconds,
                        count           = false,
                    }
                end)
                if ok and val then produced_j_tick = produced_j_tick + val end
            end

            for proto_name, _ in pairs(stats.input_counts) do
                local ok, val = pcall(function()
                    return stats.get_flow_count{
                        name            = proto_name,
                        category        = "input",
                        precision_index = defines.flow_precision_index.five_seconds,
                        count           = false,
                    }
                end)
                if ok and val then consumed_j_tick = consumed_j_tick + val end
            end

            power.produced_kw = produced_j_tick * 60 / 1000
            power.consumed_kw = consumed_j_tick * 60 / 1000
        end

        local accumulators = surface.find_entities_filtered({
            position = center,
            radius   = radius * 2,
            type     = "accumulator",
        })
        local total_charge = 0
        for _, acc in ipairs(accumulators) do
            if acc.valid then total_charge = total_charge + (acc.energy or 0) end
        end
        power.accumulated_kj = total_charge / 1000

        if power.produced_kw > 0 then
            power.satisfaction = math.min(1.0, power.consumed_kw / power.produced_kw)
        end
    end

    -- ---- Belt segments ----
    local belts_data    = {}
    local belt_entities = surface.find_entities_filtered({
        position = center,
        radius   = radius,
        type     = {"transport-belt", "underground-belt", "splitter"},
    })
    local seg_id = 1
    for _, belt in ipairs(belt_entities) do
        if belt.valid then
            local congested = false
            local item_name = nil
            if belt.get_transport_line then
                local ok, line1 = pcall(function() return belt.get_transport_line(1) end)
                if ok and line1 then
                    local ll = line1.line_length or 0
                    congested = ll > 0 and (#line1 >= ll)
                    if #line1 > 0 then
                        local stack = line1[1]
                        if stack and stack.valid_for_read then
                            item_name = stack.name
                        end
                    end
                end
            end
            table.insert(belts_data, {
                segment_id = seg_id,
                positions  = {{x = belt.position.x, y = belt.position.y}},
                congested  = congested,
                item       = item_name,
            })
            seg_id = seg_id + 1
        end
    end

    -- ---- Inserter activity ----
    local inserter_activity = {}
    local inserters = surface.find_entities_filtered({
        position = center,
        radius   = radius,
        type     = "inserter",
    })
    for _, ins in ipairs(inserters) do
        if ins.valid and ins.unit_number then
            local active = (ins.held_stack and ins.held_stack.valid_for_read) and 1 or 0
            inserter_activity[tostring(ins.unit_number)] = active
        end
    end

    return {
        power             = power,
        belts             = belts_data,
        inserter_activity = inserter_activity,
    }
end

function fa._damaged_entities_table(player, radius)
    local damaged = {}
    local surface = player.surface
    local found   = surface.find_entities_filtered({
        position = player.position,
        radius   = radius,
    })
    for _, entity in ipairs(found) do
        if entity.valid and entity.unit_number then
            -- 2.x: get_health_ratio() removed. Compute from health / max_health.
            local ratio = nil
            if entity.health and entity.max_health and entity.max_health > 0 then
                ratio = entity.health / entity.max_health
            end
            if ratio and ratio < 1.0 and ratio > 0.0 then
                table.insert(damaged, {
                    entity_id       = entity.unit_number,
                    name            = entity.name,
                    position        = {x = entity.position.x, y = entity.position.y},
                    health_fraction = ratio,
                    observed_at     = game.tick,
                })
            end
        end
    end
    return damaged
end

function fa._threat_table()
    local player           = get_player()
    local bases            = {}
    local pollution_samples = {}
    local attack_timers    = {}
    local evolution        = 0.0

    local enemy_force = game.forces["enemy"]
    if enemy_force and player and player.valid then
        -- 2.x: evolution is per-surface
        local ok, val = pcall(function()
            return enemy_force.get_evolution_factor(player.surface)
        end)
        if ok and val then evolution = val end
    end

    if player and player.valid then
        local surface  = player.surface
        local spawners = surface.find_entities_filtered({
            position = player.position,
            radius   = 512,
            type     = "unit-spawner",
            force    = "enemy",
        })
        for _, spawner in ipairs(spawners) do
            if spawner.valid and spawner.unit_number then
                -- 2.x: spawner_data removed. unit_count gives active unit count.
                local unit_count = 0
                local ok, n = pcall(function() return spawner.unit_count end)
                if ok and n then unit_count = n end

                table.insert(bases, {
                    base_id   = spawner.unit_number,
                    position  = {x = spawner.position.x, y = spawner.position.y},
                    size      = unit_count,
                    evolution = evolution,
                })
            end
        end
    end

    return {
        biter_bases      = bases,
        pollution_cloud  = pollution_samples,
        attack_timers    = attack_timers,
        evolution_factor = evolution,
    }
end

-- ============================================================
-- Prototype query functions
-- Called by KnowledgeBase when an unknown entity/resource/fluid/
-- recipe/technology is first encountered at runtime.
-- Each returns a JSON object with the prototype's key properties,
-- or {"ok":false,"reason":"..."} if the prototype doesn't exist.
-- ============================================================

-- Returns physical + crafting metadata for a placed entity prototype.
-- Used to populate EntityRecord on first encounter.
function fa.get_entity_prototype(entity_name)
    local proto = game.entity_prototypes[entity_name]
    if not proto then
        return err_response("unknown_entity_prototype: " .. tostring(entity_name))
    end

    -- Ingredient slot count: only assembling-machine types expose this.
    local ingredient_slots = 0
    local ok_ing, n_ing = pcall(function()
        return proto.ingredient_count or 0
    end)
    if ok_ing and n_ing then ingredient_slots = n_ing end

    -- has_recipe_slot: true if the entity has a crafting_categories table.
    local has_recipe_slot = false
    local ok_cc, cc = pcall(function() return proto.crafting_categories end)
    if ok_cc and cc and next(cc) ~= nil then has_recipe_slot = true end

    -- output_slots: approximated from the result inventory size.
    local output_slots = 0
    local ok_oi, oi = pcall(function()
        return proto.get_inventory_size(defines.inventory.assembling_machine_output)
            or proto.get_inventory_size(defines.inventory.furnace_result)
            or 0
    end)
    if ok_oi and oi then output_slots = oi end

    return safe_json({
        name             = proto.name,
        type             = proto.type,
        tile_width       = proto.tile_width  or 1,
        tile_height      = proto.tile_height or 1,
        has_recipe_slot  = has_recipe_slot,
        ingredient_slots = ingredient_slots,
        output_slots     = output_slots,
    })
end

-- Returns fluid prototype properties.
-- The temperature argument is informational only (used as a label by Python);
-- the prototype data is the same regardless of temperature variant.
function fa.get_fluid_prototype(fluid_name)
    local proto = game.fluid_prototypes[fluid_name]
    if not proto then
        return err_response("unknown_fluid_prototype: " .. tostring(fluid_name))
    end

    -- fuel_value is in joules (J); Python converts to MJ.
    local fuel_value = 0.0
    local ok_fv, fv = pcall(function() return proto.fuel_value or 0.0 end)
    if ok_fv and fv then fuel_value = fv end

    return safe_json({
        name                  = proto.name,
        default_temperature   = proto.default_temperature or 15,
        max_temperature       = proto.max_temperature or proto.default_temperature or 15,
        fuel_value            = fuel_value,
        emissions_multiplier  = proto.emissions_multiplier or 1.0,
        is_hidden             = proto.hidden or false,
    })
end

-- Returns resource patch prototype properties (mineable resources).
function fa.get_resource_prototype(resource_name)
    local proto = game.entity_prototypes[resource_name]
    if not proto or proto.type ~= "resource" then
        return err_response("unknown_resource_prototype: " .. tostring(resource_name))
    end

    -- Determine if the resource yields a fluid.
    local is_fluid = false
    local is_infinite = false
    local ok_mp, mp = pcall(function() return proto.mineable_properties end)
    if ok_mp and mp then
        is_infinite = mp.infinite or false
        for _, product in ipairs(mp.products or {}) do
            if product.type == "fluid" then
                is_fluid = true
                break
            end
        end
    end

    -- Build a display name from the localised name if available, else the raw name.
    local display_name = resource_name
    local ok_ln, ln = pcall(function()
        return proto.localised_name and helpers.localise_string(proto.localised_name)
    end)
    if ok_ln and ln then display_name = ln end

    return safe_json({
        name         = proto.name,
        is_fluid     = is_fluid,
        is_infinite  = is_infinite,
        display_name = display_name,
    })
end

-- Returns recipe prototype: ingredients, products, crafting time, category.
-- Called when KnowledgeBase.ensure_recipe() encounters an unknown recipe.
function fa.get_recipe_prototype(recipe_name)
    -- Recipes are force-specific in 2.x; use the player force.
    local force = game.forces["player"]
    local recipe = force and force.recipes[recipe_name]

    -- Fall back to the raw prototype if the force recipe isn't available
    -- (e.g. during early game before the recipe is enabled).
    if not recipe then
        local proto = game.recipe_prototypes[recipe_name]
        if not proto then
            return err_response("unknown_recipe: " .. tostring(recipe_name))
        end

        local ingredients = {}
        for _, ing in ipairs(proto.ingredients or {}) do
            table.insert(ingredients, {
                name        = ing.name,
                amount      = ing.amount or 1,
                type        = ing.type or "item",
                temperature = ing.temperature,
            })
        end
        local products = {}
        for _, prod in ipairs(proto.products or {}) do
            table.insert(products, {
                name        = prod.name,
                amount      = prod.amount or prod.amount_min or 1,
                probability = prod.probability or 1.0,
                type        = prod.type or "item",
                temperature = prod.temperature,
            })
        end

        -- Determine which entity types can craft this recipe category.
        local made_in = {}
        for proto_name, ep in pairs(game.entity_prototypes) do
            local ok, cc = pcall(function() return ep.crafting_categories end)
            if ok and cc and cc[proto.category] then
                table.insert(made_in, proto_name)
            end
        end

        return safe_json({
            name             = proto.name,
            category         = proto.category or "crafting",
            energy_required  = proto.energy_required or 0.5,
            ingredients      = ingredients,
            products         = products,
            made_in          = made_in,
            enabled          = proto.enabled,
        })
    end

    -- Use the force recipe (has live enabled state).
    local ingredients = {}
    for _, ing in ipairs(recipe.ingredients or {}) do
        table.insert(ingredients, {
            name        = ing.name,
            amount      = ing.amount or 1,
            type        = ing.type or "item",
            temperature = ing.temperature,
        })
    end
    local products = {}
    for _, prod in ipairs(recipe.products or {}) do
        table.insert(products, {
            name        = prod.name,
            amount      = prod.amount or prod.amount_min or 1,
            probability = prod.probability or 1.0,
            type        = prod.type or "item",
            temperature = prod.temperature,
        })
    end

    local made_in = {}
    for proto_name, ep in pairs(game.entity_prototypes) do
        local ok, cc = pcall(function() return ep.crafting_categories end)
        if ok and cc and cc[recipe.category] then
            table.insert(made_in, proto_name)
        end
    end

    return safe_json({
        name             = recipe.name,
        category         = recipe.category or "crafting",
        energy_required  = recipe.energy,
        ingredients      = ingredients,
        products         = products,
        made_in          = made_in,
        enabled          = recipe.enabled,
    })
end

-- Returns technology node: prerequisites and unlock effects.
-- Called when KnowledgeBase.ensure_tech() encounters an unknown tech.
function fa.get_technology(tech_name)
    local force = game.forces["player"]
    if not force then return err_response("no_player_force") end

    local tech = force.technologies[tech_name]
    if not tech then
        return err_response("unknown_technology: " .. tostring(tech_name))
    end

    local prerequisites = {}
    for name, _ in pairs(tech.prerequisites or {}) do
        table.insert(prerequisites, name)
    end

    local effects = {}
    for _, effect in ipairs(tech.effects or {}) do
        table.insert(effects, {
            type   = effect.type,
            recipe = effect.recipe,
            item   = effect.item,
        })
    end

    return safe_json({
        name          = tech.name,
        prerequisites = prerequisites,
        effects       = effects,
        researched    = tech.researched,
        enabled       = tech.enabled,
    })
end

-- ============================================================
-- Action command functions
-- ============================================================

local function ok_response()
    return '{"ok":true}'
end

local function err_response(reason)
    return helpers.table_to_json({ok = false, reason = reason})
end

-- 2.x: walking_state is on player.character, not the player object directly.
-- Supports all 8 compass directions; uses a diagonal when the angle to the
-- target is within 45° of a diagonal (2:1 ratio threshold on axis displacement).
function fa.move_to(position, pathfind)
    local player = get_player()
    if not player or not player.valid or not player.character then
        return err_response("no_character")
    end

    local dx = position.x - player.position.x
    local dy = position.y - player.position.y

    -- Close enough — halt and return success so the execution layer stops looping.
    if math.abs(dx) < 0.5 and math.abs(dy) < 0.5 then
        player.character.walking_state = {
            walking   = false,
            direction = defines.direction.north,
        }
        return ok_response()
    end

    if not pathfind then
        player.teleport({x = position.x, y = position.y})
        return ok_response()
    end

    local ax, ay = math.abs(dx), math.abs(dy)
    local dir
    if ax > ay * 2 then
        dir = dx > 0 and defines.direction.east or defines.direction.west
    elseif ay > ax * 2 then
        dir = dy > 0 and defines.direction.south or defines.direction.north
    elseif dx > 0 and dy > 0 then
        dir = defines.direction.southeast
    elseif dx > 0 and dy < 0 then
        dir = defines.direction.northeast
    elseif dx < 0 and dy > 0 then
        dir = defines.direction.southwest
    else
        dir = defines.direction.northwest
    end

    player.character.walking_state = {walking = true, direction = dir}
    return ok_response()
end

function fa.stop_movement()
    local player = get_player()
    if not player or not player.valid or not player.character then
        return err_response("no_character")
    end
    player.character.walking_state = {walking = false, direction = defines.direction.north}
    return ok_response()
end

function fa.mine_resource(position, resource_name, count)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end
    local surface = player.surface

    -- If resource_name is empty or nil, find whatever resource is at the
    -- position. This allows the Python side to issue a positional mining
    -- command without knowing the resource type in advance.
    local filter = {
        position = position,
        radius   = 1.5,
        type     = "resource",
    }
    if resource_name and resource_name ~= "" then
        filter.name = resource_name
    end

    local resources = surface.find_entities_filtered(filter)
    if #resources == 0 then return err_response("no_resource_at_position") end

    local target = resources[1]
    local reach  = (player.character and player.character.reach_distance) or 6
    local dist   = math.sqrt(
        (player.position.x - target.position.x)^2 +
        (player.position.y - target.position.y)^2
    )
    if dist > reach + 2 then return err_response("out_of_reach") end

    player.update_selected_entity(target.position)
    if player.selected and player.selected.valid then
        player.mining_state = {mining = true, position = target.position}
    else
        return err_response("cannot_select_resource")
    end
    return ok_response()
end

function fa.mine_entity(entity_id)
    local player  = get_player()
    if not player or not player.valid then return err_response("no_player") end
    local surface = player.surface
    local reach   = (player.character and player.character.reach_distance) or 6

    local target = nil
    local candidates = surface.find_entities_filtered({
        position = player.position,
        radius   = reach + 2,
    })
    for _, e in ipairs(candidates) do
        if e.unit_number == entity_id then
            target = e
            break
        end
    end
    if not target or not target.valid then return err_response("entity_not_found") end

    local dist = math.sqrt(
        (player.position.x - target.position.x)^2 +
        (player.position.y - target.position.y)^2
    )
    if dist > reach + 2 then return err_response("out_of_reach") end

    player.update_selected_entity(target.position)
    player.mining_state = {mining = true, position = target.position}
    return ok_response()
end

function fa.craft_item(recipe_name, count)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end
    local queued = player.begin_crafting({recipe = recipe_name, count = count})
    if queued == 0 then
        return err_response("cannot_craft_missing_ingredients_or_unknown_recipe")
    end
    return safe_json({ok = true, queued = queued})
end

function fa.place_entity(item_name, position, direction)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end

    if player.get_item_count(item_name) == 0 then
        return err_response("item_not_in_inventory")
    end

    local placed = player.surface.create_entity({
        name        = item_name,
        position    = position,
        direction   = direction or defines.direction.north,
        force       = player.force,
        player      = player,
        raise_built = true,
    })
    if not placed then
        return err_response("placement_failed_collision_or_invalid_position")
    end
    player.remove_item({name = item_name, count = 1})
    return safe_json({ok = true, unit_number = placed.unit_number})
end

function fa.set_recipe(entity_id, recipe_name)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end

    local entity = fa._find_entity_by_id(player, entity_id)
    if not entity             then return err_response("entity_not_found") end
    if not entity.set_recipe  then return err_response("entity_has_no_recipe_slot") end

    local ok, err = pcall(function() entity.set_recipe(recipe_name) end)
    if not ok then return err_response("invalid_recipe: " .. tostring(err)) end
    return ok_response()
end

function fa.set_filter(entity_id, slot, item_name)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end

    local entity = fa._find_entity_by_id(player, entity_id)
    if not entity            then return err_response("entity_not_found") end
    if not entity.set_filter then return err_response("entity_has_no_filter") end

    local lua_slot = slot + 1   -- Python 0-indexed → Lua 1-indexed
    local ok, err  = pcall(function()
        if item_name and item_name ~= "" then
            entity.set_filter(lua_slot, item_name)
        else
            entity.set_filter(lua_slot, nil)
        end
    end)
    if not ok then return err_response("set_filter_failed: " .. tostring(err)) end
    return ok_response()
end

function fa.apply_blueprint(blueprint_string, position, direction, force_build)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end

    local bp = player.cursor_stack
    if not bp then return err_response("no_cursor_stack") end

    bp.set_stack({name = "blueprint"})
    local import_result = bp.import_stack(blueprint_string)
    if import_result ~= 0 then
        bp.clear()
        return err_response("invalid_blueprint_string code=" .. tostring(import_result))
    end

    local entities = bp.get_blueprint_entities()
    if not entities then
        bp.clear()
        return err_response("blueprint_has_no_entities")
    end

    local build_result = bp.build_blueprint({
        surface     = player.surface,
        force       = player.force,
        position    = position,
        direction   = direction or defines.direction.north,
        force_build = force_build or false,
    })
    bp.clear()
    if not build_result then return err_response("blueprint_build_failed") end
    return safe_json({ok = true, built = #build_result})
end

function fa.transfer_items(entity_id, to_player, item_name, count)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end

    local entity = fa._find_entity_by_id(player, entity_id)
    if not entity then return err_response("entity_not_found") end

    local entity_inv = entity.get_inventory(defines.inventory.chest)
                    or entity.get_inventory(defines.inventory.assembling_machine_output)
                    or entity.get_inventory(defines.inventory.furnace_result)
    if not entity_inv then return err_response("entity_has_no_inventory") end

    local player_inv = player.get_inventory(defines.inventory.character_main)
    if not player_inv then return err_response("no_player_inventory") end

    if to_player then
        local moved = 0
        if item_name then
            moved = entity_inv.remove({name = item_name, count = count})
            if moved > 0 then
                player_inv.insert({name = item_name, count = moved})
            end
        else
            for i = 1, #entity_inv do
                local stack = entity_inv[i]
                if stack and stack.valid_for_read then
                    local n = player_inv.insert(stack)
                    entity_inv.remove({name = stack.name, count = n})
                    moved = moved + n
                end
            end
        end
        return safe_json({ok = true, transferred = moved})
    else
        if not item_name then
            return err_response("item_name_required_for_player_to_entity")
        end
        local available = player.get_item_count(item_name)
        local to_move   = (count > 0) and math.min(count, available) or available
        if to_move == 0 then return err_response("item_not_in_inventory") end
        local moved = entity_inv.insert({name = item_name, count = to_move})
        player.remove_item({name = item_name, count = moved})
        return safe_json({ok = true, transferred = moved})
    end
end

function fa.set_research_queue(technologies)
    local force = game.forces["player"]
    if not force then return err_response("no_player_force") end

    -- 2.x: research_queue is not directly assignable. Clear by cancelling the
    -- current research (which also clears the rest of the queue in 2.x),
    -- then re-enqueue using add_research().
    if force.current_research then
        force.cancel_current_research()
    end

    local added = {}
    for _, tech_name in ipairs(technologies) do
        local tech = force.technologies[tech_name]
        if tech and not tech.researched then
            local ok = pcall(function() force.add_research(tech_name) end)
            if ok then table.insert(added, tech_name) end
        end
    end
    return safe_json({ok = true, queued = added})
end

function fa.equip_armor(item_name)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end
    if player.get_item_count(item_name) == 0 then
        return err_response("item_not_in_inventory")
    end
    local armor_inv = player.get_inventory(defines.inventory.character_armor)
    if not armor_inv then return err_response("no_armor_slot") end
    local inserted = armor_inv.insert({name = item_name, count = 1})
    if inserted == 0 then
        return err_response("armor_slot_occupied_or_wrong_type")
    end
    player.remove_item({name = item_name, count = 1})
    return ok_response()
end

function fa.use_item(item_name, target_position)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end
    if player.get_item_count(item_name) == 0 then
        return err_response("item_not_in_inventory")
    end
    player.cursor_stack.set_stack({name = item_name, count = 1})
    if target_position then
        player.surface.create_entity({
            name     = item_name,
            position = target_position,
            player   = player,
            force    = player.force,
        })
    end
    player.remove_item({name = item_name, count = 1})
    player.cursor_stack.clear()
    return ok_response()
end

function fa.rotate_entity(entity_id, reverse)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end

    local entity = fa._find_entity_by_id(player, entity_id)
    if not entity then return err_response("entity_not_found") end

    -- entity.rotate() returns true on success, false if the entity does not
    -- support rotation (e.g. chests, accumulators). Guard with pcall in case
    -- the prototype does not expose the method at all.
    local ok, result = pcall(function()
        return entity.rotate({reverse = reverse or false})
    end)

    if not ok then
        return err_response("rotate_failed: " .. tostring(result))
    end
    if not result then
        return err_response("rotate_not_supported")
    end
    return ok_response()
end

function fa.flip_entity(entity_id, horizontal)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end

    local entity = fa._find_entity_by_id(player, entity_id)
    if not entity then return err_response("entity_not_found") end

    -- entity.flip() is available in Factorio 2.x for entities that support
    -- mirroring (oil refineries, chemical plants, etc.). Returns true on
    -- success, false when the entity type does not allow flipping.
    -- horizontal=true mirrors left<->right; horizontal=false mirrors top<->bottom.
    local ok, result = pcall(function()
        return entity.flip(horizontal)
    end)

    if not ok then
        return err_response("flip_failed: " .. tostring(result))
    end
    if not result then
        return err_response("flip_not_supported")
    end
    return ok_response()
end

function fa.set_splitter_priority(entity_id, input_priority, output_priority)
    local player = get_player()
    if not player or not player.valid then return err_response("no_player") end

    local entity = fa._find_entity_by_id(player, entity_id)
    if not entity then return err_response("entity_not_found") end

    -- Verify this is a splitter — only splitters expose input_priority /
    -- output_priority properties in the Factorio API.
    if entity.type ~= "splitter" then
        return err_response("entity_is_not_a_splitter")
    end

    -- Each priority field is set independently so that passing nil for one
    -- leaves the existing setting unchanged.
    local ok, err = pcall(function()
        if input_priority ~= nil then
            entity.input_priority = input_priority
        end
        if output_priority ~= nil then
            entity.output_priority = output_priority
        end
    end)

    if not ok then
        return err_response("set_splitter_priority_failed: " .. tostring(err))
    end
    return ok_response()
end

-- ============================================================
-- VEHICLE stubs
-- ============================================================

function fa.enter_vehicle(entity_id)
    return err_response("vehicle_actions_not_implemented")
end

function fa.exit_vehicle()
    return err_response("vehicle_actions_not_implemented")
end

function fa.drive_vehicle(position, pathfind)
    return err_response("vehicle_actions_not_implemented")
end

-- ============================================================
-- COMBAT stubs
-- ============================================================

function fa.select_weapon(slot)
    return err_response("combat_actions_not_implemented")
end

function fa.shoot_at(target_entity_id, target_position)
    return err_response("combat_actions_not_implemented")
end

function fa.stop_shooting()
    return err_response("combat_actions_not_implemented")
end

-- ============================================================
-- Internal utility: find entity by unit_number within reach
-- ============================================================

function fa._find_entity_by_id(player, entity_id)
    local reach      = (player.character and player.character.reach_distance) or 6
    local candidates = player.surface.find_entities_filtered({
        position = player.position,
        radius   = reach + 4,
    })
    for _, e in ipairs(candidates) do
        if e.unit_number == entity_id then return e end
    end
    return nil
end


-- Testing: load suite and register as a remote command
T = require("test_bridge_live")

commands.add_command(
    "agent-test",
    "Run the factorio-agent bridge test suite. Usage: /agent-test [suite_name]",
    function(event)
        local arg = event.parameter
        if arg and arg ~= "" then
            T.run_suite(arg)
        else
            T.run_all()
        end
    end
)