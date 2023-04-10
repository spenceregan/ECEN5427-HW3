
using CSV, LinearAlgebra, DataFrames, PrettyTables
using JuMP, Gurobi
root = dirname(@__FILE__)

plants_path = joinpath(root, "plant_characteristics.csv")
timeseries_path = joinpath(root, "timeseries.csv")
plants = CSV.read(plants_path, DataFrame)
plants.lifetime = [30,30,60,30,30]
sdr = 0.03 #social discount rate
pmt(capex, yrs) = sdr*capex/(1-(1+sdr)^(-yrs))
transform!(plants, [:cap_cost, :lifetime] => ByRow(pmt) => :ann_capex)
timeseries = CSV.read(timeseries_path, DataFrame)

function capacity_expansion(
    plants::DataFrame,
    timeseries::DataFrame
    )
    plant_types = plants.plant_char

    timestep_hours = 1.0
    timeseries_years = (timestep_hours * nrow(timeseries))/8760

    n = nrow(plants)
    s = nrow(timeseries)
    σ = plants.ann_capex * 1000 * timeseries_years
    ξ = plants.fom_cost * 1000 * timeseries_years
    χ = plants.vom_cost
    υ = plants.fuel_cost
    d = timeseries.load_MW

    A = ones((s,n))
    A[:, 4] = timeseries.solar_cf
    A[:, 5] = timeseries.wind_cf

    capacityexp = Model(Gurobi.Optimizer)

    @variable(capacityexp, γ[1:n] ≥ 0)
    @variable(capacityexp, P[1:s, 1:n] ≥ 0)

    @objective(capacityexp, Min, (σ + ξ)'*γ + ones(s)'*P*(χ + υ))

    @constraint(capacityexp, P .≤ A * Diagonal(γ))
    @constraint(capacityexp, P*ones(n) .== d)

    optimize!(capacityexp)
    optimal_fleet = DataFrame(plant = plant_types, Capacity_GW = value.(γ) / 1000.0)
    optimal_cost = objective_value(capacityexp)
    return optimal_fleet, optimal_cost
end


optimal_fleet, optimal_cost = capacity_expansion(plants, timeseries)


println("Optimal annual cost = \$ ", round(optimal_cost/1e9; digits=1), " B")


pretty_table(optimal_fleet, nosubheader = true)


ts_jan = timeseries[1:720, :]
fleet_jan, cost_reduced = capacity_expansion(plants, ts_jan)


pretty_table(fleet_jan, nosubheader = true)


ts_summer = timeseries[5000:5720, :]
fleet_summer, cost_reduced = capacity_expansion(plants, ts_summer)


pretty_table(fleet_summer, nosubheader = true)


function capacity_expansion_CO2(
    plants::DataFrame,
    timeseries::DataFrame,
    ts_cap::AbstractFloat
    )
    plant_types = plants.plant_char

    timestep_hours = 1.0
    timeseries_years = (timestep_hours * nrow(timeseries))/8760

    c = ts_cap

    n = nrow(plants)
    s = nrow(timeseries)
    σ = plants.ann_capex * 1000 * timeseries_years
    ξ = plants.fom_cost * 1000 * timeseries_years
    ρ = plants.emit_rate
    χ = plants.vom_cost
    υ = plants.fuel_cost
    d = timeseries.load_MW

    A = ones((s,n))
    A[:, 4] = timeseries.solar_cf
    A[:, 5] = timeseries.wind_cf

    capacityexp = Model(Gurobi.Optimizer)

    @variable(capacityexp, γ[1:n] ≥ 0)
    @variable(capacityexp, P[1:s, 1:n] ≥ 0)

    @objective(capacityexp, Min, (σ + ξ)'*γ + ones(s)'*P*(χ + υ))

    @constraint(capacityexp, P .≤ A * Diagonal(γ))
    @constraint(capacityexp, pbal, P*ones(n) .== d)
    @constraint(capacityexp, ccap, ones(s)'*P*ρ ≤ c)

    optimize!(capacityexp)
    optimal_fleet = DataFrame(plant = plant_types, Capacity_GW = value.(γ) / 1000.0)
    optimal_cost = objective_value(capacityexp)
    cprice = dual(ccap)
    return optimal_fleet, optimal_cost, cprice
end


fleet_CO2, cost_CO2, cprice = capacity_expansion_CO2(plants, timeseries, 100e6)


println("Optimal annual cost with 100 MMT annual emmissions cap = \$ ", round(cost_CO2/1e9; digits=1), " B")
pretty_table(fleet_CO2, nosubheader = true)


println("Shadow carbon price = \$", round(-1*cprice; digits = 2), " per ton")

