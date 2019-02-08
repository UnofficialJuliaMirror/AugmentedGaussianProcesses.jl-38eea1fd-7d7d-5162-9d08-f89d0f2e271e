"""
Softmax likelihood : ``p(y=i|{fₖ}) = exp(fᵢ)/ ∑ exp(fₖ) ``
"""
abstract type AbstractLogisticSoftMaxLikelihood{T<:Real} <: MultiClassLikelihood{T} end

struct AugmentedLogisticSoftMaxLikelihood{T<:Real} <: AbstractLogisticSoftMaxLikelihood{T}
    Y::AbstractVector{SparseVector{Int64}} #Mapping from instances to classes
    class_mapping::AbstractVector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    y_class::AbstractVector{Int64} # GP Index for each sample
    c::AbstractVector{AbstractVector{T}} # Second moment of fₖ
    α::AbstractVector{T} # Variational parameter of Gamma distribution
    β::AbstractVector{T} # Variational parameter of Gamma distribution
    θ::AbstractVector{AbstractVector{T}} # Variational parameter of Polya-Gamma distribution
    γ::AbstractVector{AbstractVector{T}} # Variational parameter of Poisson distribution
    function AugmentedLogisticSoftMaxLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function AugmentedLogisticSoftMaxLikelihood{T}(Y::AbstractVector{SparseVector{<:Int}},
    class_mapping::AbstractVector{Any}, ind_mapping::Dict{Any,Int},y_class::AbstractVector{<:Int}) where {T<:Real}
        new{T}(Y,class_mapping,ind_mapping,y_class)
    end
    function AugmentedLogisticSoftMaxLikelihood{T}(Y::AbstractVector{SparseVector{<:Integer}},
    class_mapping::AbstractVector{Any}, ind_mapping::Dict{Any,Int},y_class::AbstractVector{<:Int},
    c::AbstractVector{AbstractVector{T}}, α::AbstractVector{T},
    β::AbstractVector{T}, θ::AbstractVector{AbstractVector{T}},γ::AbstractVector{AbstractVector{T}}) where {T<:Real}
        new{T}(Y,class_mapping,ind_mapping,c,α,β,θ,γ)
    end
end

function AugmentedLogisticSoftMaxLikelihood()
    AugmentedLogisticSoftMaxLikelihood{Float64}()
end

isaugmented(::AugmentedLogisticSoftMaxLikelihood{T}) where T = true

function pdf(l::AbstractLogisticSoftMaxLikelihood,f::AbstractVector)
    logisticsoftmax(f)
end

function init_likelihood(likelihood::AugmentedLogisticSoftMaxLikelihood{T},nLatent::Integer,nSamplesUsed::Integer) where T
    c = [ones(T,nSamplesUsed) for i in 1:nLatent]
    α = nLatent*ones(T,nSamplesUsed)
    β = nLatent*ones(T,nSamplesUsed)
    θ = [abs.(rand(T,nSamplesUsed))*2 for i in 1:nLatent]
    γ = [abs.(rand(T,nSamplesUsed)) for i in 1:nLatent]
    SoftMaxLikelihood{T}(likelihood.Y,likelihood.class_mapping,likelihood.ind_mapping,c,α,β,θ,γ)
end

struct LogisticSoftMaxLikelihood{T<:Real} <: AbstractLogisticSoftMaxLikelihood{T}
    Y::AbstractVector{SparseVector{Int64}} #Mapping from instances to classes
    class_mapping::AbstractVector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    y_class::AbstractVector{Int64} #GP Index for each sample
    function LogisticSoftMaxLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function LogisticSoftMaxLikelihood{T}(Y::AbstractVector{SparseVector{<:Integer}},
    class_mapping::AbstractVector{Any}, ind_mapping::Dict{Any,Int},y_class::AbstractVector{<:Int}) where {T<:Real}
        new{T}(Y,class_mapping,ind_mapping,y_class)
    end
end

function LogisticSoftMaxLikelihood()
    LogisticSoftMaxLikelihood{Float64}()
end

function init_likelihood(likelihood::LogisticSoftMaxLikelihood{T},nLatent::Integer,nSamplesUsed::Integer) where T
    return likelihood
end

function local_updates!(model::VGP{<:AugmentedLogisticSoftMaxLikelihood,<:AnalyticInference})
    model.likelihood.c .= broadcast((Σ,μ)->sqrt.(Σ.+μ.^2),diag.(model.Σ),model.μ)
    for _ in 1:2
        model.likelihood.γ .= broadcast((c,μ)->0.5./(model.likelihood.β.*cosh.(0.5.*c)).*exp.(digamma.(model.likelihood.α).-0.5.*μ),
                                    model.likelihood.c,model.μ)
        model.likelihood.α .= [1.0+sum(γ[i] for γ in model.likelihood.γ) for i in 1:model.nSamples]
    end
    model.likelihood.θ .= broadcast((y,γ,c)->0.5.*Array(y+γ)./c.*tanh.(0.5.*c),model.likelihood.Y,model.likelihood.γ,model.likelihood.c)
    model.inference.∇μE .= 0.5.*Array.(model.likelihood.Y-model.likelihood.γ)
    model.inference.∇ΣE .= 0.5.*model.likelihood.θ
end

function local_updates!(model::SVGP{<:AugmentedLogisticSoftMaxLikelihood,<:AnalyticInference})
    model.likelihood.c .= broadcast((μ::AbstractVector,Σ::AbstractMatrix,κ::AbstractMatrix,K̃::AbstractVector)->sqrt.(K̃+opt_diag(κ*Σ,κ)+(κ*μ).^2),
                                    model.μ,model.Σ,model.κ,model.K̃)
    for _ in 1:10
        model.likelihood.γ .= broadcast((c,κ,μ)->0.5./(model.likelihood.β.*cosh.(0.5.*c)) .*exp.(digamma.(model.likelihood.α).-0.5.*κ*μ),
                                    model.likelihood.c,model.κ,model.μ)
        model.likelihood.α .= [1.0+sum(γ[i] for γ in model.likelihood.γ) for i in 1:model.nSamplesUsed]
    end
    model.likelihood.θ .= broadcast((y,γ::Vector,c::Vector)->0.5.*Array(y[model.inference.MBIndices]+γ)./c.*tanh.(0.5.*c),
                                    model.likelihood.Y,model.likelihood.γ,model.likelihood.c)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:AugmentedLogisticSoftMaxLikelihood},index::Integer)
    0.5.*Array(model.likelihood.Y[index]-model.likelihood.γ[index])
end

function expec_μ(model::VGP{<:AugmentedLogisticSoftMaxLikelihood})
    0.5.*Array.(model.likelihood.Y-model.likelihood.γ)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:AugmentedLogisticSoftMaxLikelihood},index::Integer)
    0.5*model.likelihood.ρ*Array(model.likelihood.Y[index][model.MBIndices]-model.likelihood.γ[index])
end

function expec_μ(model::SVGP{<:AugmentedLogisticSoftMaxLikelihood})
    0.5*model.likelihood.ρ*Array.(getindex.(model.likelihood.Y,[model.MBIndices]).-model.likelihood.γ)
end

function expec_Σ(model::GP{<:AugmentedLogisticSoftMaxLikelihood},index::Integer)
    0.5*model.likelihood.θ[index]
end

function expec_Σ(model::GP{<:AugmentedLogisticSoftMaxLikelihood})
    0.5.*model.likelihood.θ
end

function compute_proba(l::AbstractLogisticSoftMaxLikelihood{T},μ::AbstractVector{AbstractVector},σ²::AbstractVector{AbstractVector}) where T
    n = length(μ[1])
    μ = hcat(μ...)
    μ = [μ[i,:] for i in 1:n]
    σ² = hcat(σ²...)
    σ² = [σ²[i,:] for i in 1:n]
    pred = zeros(n,length(model.Y))
    nSamples = 200
    for i in 1:n
        p = MvNormal(m_f[i],sqrt.(max.(eps(T),cov_f[i])))
        for _ in 1:nSamples
            pred[i,:] += pdf(l,rand(p))/nSamples
        end
    end
    return DataFrame(pred,Symbol.(l.class_mapping))
end


function ELBO(model::GP{<:AugmentedLogisticSoftMaxLikelihood})
    return expecLogLikelihood(model) - GaussianKL(model) - GammaImproperKL(model) - PoissonKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{<:AugmentedLogisticSoftMaxLikelihood})
    tot = -model.nSamples*log(2)
    tot += -sum(sum.(model.likelihood.γ))*log(2.0)
    tot +=  0.5*sum(broadcast((y,μ,γ,θ,c)->sum(μ.*Array(y-γ)-θ.*(c.^2)),
                    model.likelihood.Y,model.μ,model.likelihood.γ,model.likelihood.θ,model.likelihood.c))
    return tot
end

function expecLogLikelihood(model::SVGP{<:AugmentedLogisticSoftMaxLikelihood})
    tot = -model.nSamplesUsed*log(2.0)
    tot += -sum(sum.(model.likelihood.γ))*log(2.0)
    tot += 0.5*sum(broadcast((y,κ,μ,γ,θ,c)->sum((κ*μ).*Array(y[model.inference.MBIndices]-γ)-θ.*(c.^2)),
                    model.likelihood.Y,model.likelihood.κ,model.μ,model.likelihood.γ,model.likelihood.θ,model.likelihood.c))
    return model.inference.ρ*tot
end

function treat_samples(model::GP{<:LogisticSoftMaxLikelihood},samples::AbstractMatrix,index::Integer)
    class = model.likelihood.ind_mapping[model.y[index]]
    grad_μ = zeros(model.nLatent)
    grad_Σ = zeros(model.nLatent)
    for i in 1:size(samples,1)
        σ = logistic(samples[i,:])
        samples[i,:]  .= logisticsoftmax(samples[i,:])
        s = samples[i,class]
        g_μ = grad_logisticsoftmax(samples[i,:],σ,class)
        grad_μ .+= g_μ./s
        grad_Σ .+= diaghessian_logisticsoftmax(samples[i,:],σ,class)./s .- g_μ.^2 ./s^2
    end
    for k in 1:model.nLatent
        model.inference.∇μE[k][index] = grad_μ[k]/nSamples
        model.inference.∇ΣE[k][index] = 0.5.*grad_Σ[k]/nSamples
    end
end

function remove_augmentation(l::Type{AugmentedLogisticSoftMaxLikelihood{T}}) where T
    return LogisticSoftMaxLikelihood{T}(l.Y,l.class_mapping,l.ind_mapping)
end

function logisticsoftmax(f::AbstractVector{<:Real})
    s = logit.(f)
    return s./sum(s)
end

function logisticsoftmax(f::AbstractVector{<:Real},i::Integer)
    return logisticsoftmax(f)[i]
end

function grad_logisticsoftmax(s::AbstractVector{<:Real},σ::AbstractVector{<:Real},i::Integer)
    base_grad = -s.*(1.0.-σ).*s[i]
    base_grad[i] += s[i]*(1.0-σ[i])
    return base_grad
end

function diaghessian_logisticsoftmax(s::AbstractVector{<:Real},σ::AbstractVector{<:Real},i::Integer)
    m = length(s)
    hessian = zeros(m)
    for j in 1:m
            hessian[j] = (1-σ[j])*s[i]*(
            (δ(i,j)-s[j])*(1.0-σ[j])*(δ(i,j)-s[j])
            -s[j]*(1.0-s[j])*(1.0-σ[j])
            -σ[j]*(δ(i,j)-s[j]))
    end
    return hessian
end

function hessian_logisticsoftmax(s::AbstractVector{<:Real},σ::AbstractVector{<:Real},i::Integer)
    m = length(s)
    hessian = zeros(m,m)
    for j in 1:m
        for k in 1:m
            hessian[j,k] = (1-σ[j])*s[i]*(
            (δ(i,k)-s[k])*(1.0-σ[k])*(δ(i,j)-s[j])
            -s[j]*(δ(j,k)-s[k])*(1.0-σ[k])
            -δ(k,j)*σ[j]*(δ(i,j)-s[j]))
        end
    end
    return hessian
end
