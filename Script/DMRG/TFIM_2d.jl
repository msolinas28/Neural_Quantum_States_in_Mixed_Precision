using ITensors, ITensorMPS, CSV, DataFrames

function main()
    if length(ARGS) < 3
        println("Usage: julia TFIM_2d.jl <L> <J> <h> <PBC>")
        exit(1)
    end

    L = parse(Int, ARGS[1])  
    J = parse(Float64, ARGS[2])
    h = parse(Float64, ARGS[3])
    PBC = parse(Bool, ARGS[4])

    sigx = [0 1; 1 0]
    sigz = [1 0; 0 -1]

    sites = siteinds("S=1/2", L * L)

    os = OpSum()

    if !PBC
        for i in 1:L
            for j in 1:L
                site = (i - 1) * L + j
                if j < L  
                    os += -J, sigz, site, sigz, site + 1
                end
                if i < L 
                    os += -J, sigz, site, sigz, site + L
                end
            end
        end
    else
        for i in 1:L
            for j in 1:L
                site = (i - 1) * L + j
                neighbor_right = (j < L) ? site + 1 : site - (L - 1)
                os += -J, sigz, site, sigz, neighbor_right
                neighbor_down = (i < L) ? site + L : j  
                os += -J, sigz, site, sigz, neighbor_down
            end
        end
    end
    for j in 1:L*L
        os += -h, sigx, j
    end

    H = MPO(os, sites)

    nsweeps = 10
    maxdim = [200, 200, 200, 200, 200, 800, 800, 800, 800, 800]
    cutoff = [1E-15]

    psi0 = randomMPS(sites, 2)

    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

    new_result = DataFrame(L = [L], J = [J], h = [h], energy = [energy])

    file_path = "../Data/DMRG/Energy/TFIM_2dim_PBC$(PBC).csv"

    if isfile(file_path)
        existing_df = CSV.read(file_path, DataFrame)
        combined_df = vcat(existing_df, new_result)
        CSV.write(file_path, combined_df)
    else
        CSV.write(file_path, new_result)
    end
 
    H_squared_exp = inner(psi, H, applyMPO(H, psi))               
    variance = real(H_squared_exp - energy^2)

    println("Ground state energy: $energy")
    println("Variance: $variance")
end

main()