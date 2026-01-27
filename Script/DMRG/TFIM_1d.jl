using ITensors, ITensorMPS, CSV, DataFrames

function main()
    if length(ARGS) < 3
        println("Usage: julia TFIM_1d.jl <L> <J> <h>")
        exit(1)
    end

    L = parse(Int, ARGS[1])  
    J = parse(Float64, ARGS[2])
    h = parse(Float64, ARGS[3])

    sigx = [0 1; 1 0]
    sigy = [0 -im; im 0]
    sigz = [1 0; 0 -1]

    sites = siteinds("S=1/2", L)

    H_operator = OpSum()
    for j in 1:L
        jp = j % L + 1  
        H_operator += J,sigz,j,sigz,jp
        H_operator += h,sigx,j
    end

    H = MPO(H_operator, sites)

    nsweeps = 10
    maxdim = [200, 200, 200, 200, 200, 1000, 1000, 1000, 1000, 1000]
    cutoff = [1E-15]

    psi0 = random_mps(sites; linkdims=4)

    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

    new_result = DataFrame(L = [L], J = [J], h = [h], energy = [energy])

    file_path = "../Data/DMRG/Energy/TFIM_1dim.csv"

    if isfile(file_path)
        existing_df = CSV.read(file_path, DataFrame)
        combined_df = vcat(existing_df, new_result)
        CSV.write(file_path, combined_df)
    else
        CSV.write(file_path, new_result)
    end

    println("Ground state energy: $energy")
end

main()