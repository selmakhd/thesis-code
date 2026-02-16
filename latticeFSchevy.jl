

mutable struct FSparms
    t::Float64                       ### fermion hopping
    Nx::Int64                        ### lattice length
    mu::Float64                      ### chemical potential
    k::Array{Float64,1}              ### momenta
    ek::Array{Float64,1}             ### bare dispersion in vacuum
    ξk::Array{Float64,1}             ### bare dispersion
    jks::Array{Int64,1}              ### indices of unoccupied momenta
    jps::Array{Int64,1}              ### indices of occupied momenta
    Nk::Int64                        ### number of particle states
    Np::Int64                        ### fermion number = number of hole states
    n::Float64                       ### density
    EF::Float64                      ### Fermi energy 2*pi*n^2
end

function FSparms(Nx,  mu;  t::Float64=1.0)
    
    k = 2pi/Nx*[0:Nx-1;]
    
    ek = 2t*(1 .- cos.(k))
    ξk = ek .- mu
    
    jks = [1:Nx;][ξk .> 0.0]
    jps = [1:Nx;][ξk .<= 0.0]
    Nk = length(jks)
    Np = length(jps)
    n = Np / Nx
    
    EF = 2pi*t * n^2
    
    return FSparms(t,Nx,mu,k,ek,ξk,jks,jps,Nk,Np,n,EF) 
end


mutable struct ChevyParms
    Q::Int64                ### total momentum sector
    EB::Float64             ### two body binding energy corresponding to U = (UA + UB)/2
    U::Float64              ### impurity-fermion interaction coupling
    g::Float64              ### U / N
    tX::Float64             ### impurity hopping
    eQ::Array{Float64,1}    ### impurity bare dispersion
end

function ChevyParms(Q, EB, tX, fs::FSparms)
    
    eQ = 2tX*(1 .- cos.(fs.k))
    U = fs.Nx/sum(1 ./ (-abs(EB) .- fs.ek .- eQ))
    g = U/fs.Nx
        
    return ChevyParms(Q, EB,U,g,tX,eQ)
end


function build_HChevy(fs::FSparms, cp::ChevyParms; ifterms=[1,1,1,1])
    
    H11 = zeros(1)
    H12 = zeros(fs.Nk, fs.Np)
    H22 = zeros(fs.Nk, fs.Np,   fs.Nk, fs.Np) 
    
    ### diagonal term
    if ifterms[1] == 1
    H11[1] = cp.eQ[cp.Q+1] + cp.g*fs.Np
    for (jp,p) in enumerate(fs.jps), (jk,k) in enumerate(fs.jks)
        q = mod(k-p+cp.Q, fs.Nx) + 1
        H22[jk,jp, jk,jp] += fs.ξk[k] - fs.ξk[p] + cp.eQ[q] + cp.g*fs.Np
    end
    end
    
    ### p-h creation
    if ifterms[2] == 1
    for (jp,p) in enumerate(fs.jps), (jk,k) in enumerate(fs.jks)
        H12[jk,jp] = cp.g
    end
    end
    
    ### particle scattering
    if ifterms[3] == 1
    for (jp,p) in enumerate(fs.jps)
        H22[:,jp, :,jp] .+= cp.g
    end
    end
    
    ### hole scattering
    if ifterms[4] == 1
    for (jk,k) in enumerate(fs.jks)
        H22[jk,:, jk,:] .-= cp.g
    end
    end
    
    
    H12 = reshape(H12, (fs.Nk*fs.Np))
    H22 = reshape(H22, (fs.Nk*fs.Np, fs.Nk*fs.Np))
    
    H = vcat(hcat(H11, H12'), hcat(H12,H22))
    
    @assert maximum(abs.(H .- H')) < 1e-12
        
    return Hermitian(H)
end

function assemble_spectrum(E, V, ws, gamma; justDOS=false)

    A_w = 0*ws

    for (iw,w) in enumerate(ws)
        Z = abs.(V[1,:]).^2
        if justDOS
            Z .= 1
        end
        pi_delta_w = gamma * Z ./ ((w .- E).^2 .+ gamma.^2)
        A_w[iw] = 0.5*sum(pi_delta_w)
    end
    
    dw = ws[2]-ws[1]
    Aw = 2*pi/dw * A_w ./ sum(A_w)

    return Aw
end
