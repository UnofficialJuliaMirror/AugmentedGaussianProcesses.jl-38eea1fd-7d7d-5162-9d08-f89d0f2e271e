# Functions related to the Extreme Gaussian Process Classifier (XGPC)
# (paper currently being reviewed for AIStats 2018)

function variablesUpdate_XGPC!(model::BatchXGPC,iter)
    model.α = sqrt.(diag(model.ζ)+model.μ.^2) #Cf derivation of updates
    θs = (1.0./(2.0*model.α)).*tanh.(model.α./2.0)
    (grad_η_1,grad_η_2) = naturalGradientELBO_XGPC(θs,model.y,model.invK)
    model.η_1 = grad_η_1; model.η_2 = grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*inv(model.η_2); model.μ = model.ζ*model.η_1 #Back to the distribution parameters (needed for α updates)
end


function variablesUpdate_XGPC!(model::SparseXGPC,iter)
    model.α[model.MBIndices] = sqrt.(model.Ktilde+sum(model.κ.*transpose(model.ζ*model.κ.'),2)+(model.κ*model.μ).^2)
    θs = (1.0./(2.0*model.α[model.MBIndices])).*tanh.(model.α[model.MBIndices]./2.0)
    (grad_η_1,grad_η_2) = naturalGradientELBO_XGPC(θs,model.y[model.MBIndices],model.invKmm; κ=model.κ,stoch_coef=model.Stochastic ? model.StochCoeff : 1.0)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_η_1; model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*inv(model.η_2); model.μ = model.ζ*model.η_1 #Back to the distribution parameters (needed for α updates)
end

#Return the value of the ELBO
# function ELBO(model)
#     if isa(model,SparseXGPC)
#         hyperparameters=model.hyperparameters
#         inducingPoints = model.inducingPoints
#         return function (inducingPoints,γ,hyperparameters)
#                     ELBO_v = -model.nSamples*log(2.0)+model.m/2.0
#                     invKmm = Matrix(Symmetric(inv(CreateKernelMatrix(inducingPoints,model.kernel_functions,hyperparameters)+γ[1]*eye(model.nFeatures))))
#                     Knm = CreateKernelMatrix(model.X[model.MBIndices,:],model.kernel_functions,hyperparameters,X2=inducingPoints)
#                     κ = Knm*invKmm
#                     Ktilde = CreateDiagonalKernelMatrix(model.X[model.MBIndices,:],model.kernel_functions,hyperparameters) + model.γ[1]*ones(length(model.MBIndices)) - squeeze(sum(κ.*Knm,2),2) #diag(model.κ*transpose(Knm))
#                     model.StochCoeff = model.nSamples/model.nSamplesUsed
#                     θ = 1./(2*model.α[model.MBIndices]).*tanh.(model.α[model.MBIndices]/2.0)
#                     ELBO_v += 0.5*(logdet(model.ζ)+logdet(invKmm))
#                     ELBO_v += -0.5*(trace((invKmm+model.StochCoeff*κ'*Diagonal(θ)*κ)*(model.ζ+model.μ*transpose(model.μ))))
#                     ELBO_v += model.StochCoeff*0.5*dot(model.y[model.MBIndices],κ*model.μ)
#                     ELBO_v += -0.5*model.StochCoeff*sum(θ.*Ktilde)
#                     ELBO_v += model.StochCoeff*sum(0.5*(model.α[model.MBIndices].^2).*θ-log.(cosh.(0.5*model.α[model.MBIndices])))
#                     return -ELBO_v
#         end
#     elseif isa(model,BatchXGPC)
#         hyperparameters = model.hyperparameters
#         return function (γ,hyperparameters)
#                     ELBO_v = -model.nSamples*log(2)
#                     invK = inv(CreateKernelMatrix(model.X,model.kernel_functions,hyperparameters)+γ[1]*eye(model.nFeatures))
#                     θ = 1./(2*model.α).*tanh.(model.α/2.0)
#                     ELBO_v += 0.5*(logdet(model.ζ)+logdet(invK)-trace((invK+Diagonal(θ))*(model.ζ+model.μ*transpose(model.μ))))
#                     ELBO_v += 0.5*dot(model.y,model.μ)
#                     ELBO_v += sum(0.5*(model.α.^2).*θ-log.(cosh.(0.5*model.α)))
#                     return -ELBO_v
#         end
#     else
#         err("Error")
#     end
# end
function ELBO(model::BatchXGPC)
    ELBO_v = model.nSamples*(0.5-log.(2.0)) #Constant
    θ = 1./(2*model.α).*tanh.(model.α/2.0) #Reparametrisation of c
    ELBO_v += 0.5*(logdet(model.ζ)+logdet(model.invK)) #Logdet computations
    ELBO_v += -0.5*sum((model.invK+Diagonal(θ)).*transpose(model.ζ+model.μ*transpose(model.μ))) #Computation of the trace
    ELBO_v += 0.5*dot(model.y,model.μ)
    ELBO_v += sum(0.5*(model.α.^2).*θ-log.(cosh.(0.5*model.α)))
    return -ELBO_v
end


function ELBO(model::SparseXGPC)
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    ELBO_v = -model.nSamples*log(2)+model.m/2.0
    θ = 1./(2*model.α[model.MBIndices]).*tanh.(model.α[model.MBIndices]/2.0)
    ELBO_v += 0.5*(logdet(model.ζ)+logdet(model.invKmm))
    ELBO_v += -0.5*(sum((model.invKmm+model.StochCoeff*model.κ'*Diagonal(θ)*model.κ).*transpose(model.ζ+model.μ*transpose(model.μ))))
    ELBO_v += model.StochCoeff*0.5*dot(model.y[model.MBIndices],model.κ*model.μ)
    ELBO_v += -0.5*model.StochCoeff*dot(θ,model.Ktilde)
    ELBO_v += model.StochCoeff*sum(0.5*(model.α[model.MBIndices].^2).*θ-log.(cosh.(0.5*model.α[model.MBIndices])))
    return -ELBO_v
end

function naturalGradientELBO_XGPC(θ,y,invPrior;κ=0,stoch_coef::Float64=1.0)
    if κ == 0
        #Full batch case
        grad_1 =  0.5*y
        grad_2 = -0.5*(Diagonal(θ) + invPrior)
    else
        #Sparse case
        grad_1 =  0.5*stoch_coef*κ'*y
        grad_2 = -0.5*(stoch_coef*transpose(κ)*Diagonal(θ)*κ + invPrior)
  end
  return (grad_1,grad_2)
end


function computeHyperParametersGradients(model::SparseXGPC,iter::Integer)
    #General values used for all gradients
    B = model.μ*transpose(model.μ) + model.ζ
    Kmn = CreateKernelMatrix(model.inducingPoints,model.Kernel_function;X2=model.X[model.MBIndices,:])
    Θ = Diagonal(0.25./model.α[model.MBIndices].*tanh.(0.5*model.α[model.MBIndices]))
    dim = size(model.X,2)
    #Update of both the coefficients and hyperparameters of the kernels
    gradients_kernel_param = zeros(model.nKernels)
    gradients_kernel_coeff = zeros(model.nKernels)
    gradients_inducing_points = zeros(model.m,dim)
    for i in 1:model.nKernels
        #Compute the derivative of the kernel matrices against the kernel hyperparameter
        Jnm_param,Jnn_param,Jmm_param = model.Kernels[i].coeff.*computeJ(model,model.Kernels[i].compute_deriv)
        ι_param = (Jnm_param-model.κ*Jmm_param)*model.invKmm
        Jtilde_param = Jnn_param - sum(ι_param.*(Kmn.'),2)-sum(model.κ.*Jnm_param,2)
        V_param = model.invKmm*Jmm_param
        gradients_kernel_param[i] = 0.5*(sum( (V_param*model.invKmm - model.StochCoeff*(ι_param'*Θ*model.κ + model.κ'*Θ*ι_param)).*transpose(B)) - trace(V_param)-model.StochCoeff*dot(diag(Θ),Jtilde_param)
         + model.StochCoeff*dot(model.y[model.MBIndices],ι_param*model.μ))
        Jnm_coeff,Jnn_coeff,Jmm_coeff = computeJ(model,model.Kernels[i].compute)
        ι_coeff = (Jnm_coeff-model.κ*Jmm_coeff)*model.invKmm
        Jtilde_coeff = Jnn_coeff - sum(ι_coeff.*(Kmn.'),2) - sum(model.κ.*Jnm_coeff,2)
        V_coeff = model.invKmm*Jmm_coeff
        gradients_kernel_coeff[i] = 0.5*(sum(( V_coeff*model.invKmm - model.StochCoeff*(ι_coeff'*Θ*model.κ+model.κ'*Θ*ι_coeff)) .* transpose(B)) -trace(V_coeff)-model.StochCoeff*sum(Θ.*Jtilde_coeff))
         + model.StochCoeff*0.5*dot(model.y[model.MBIndices],ι_coeff*model.μ)
    end
    for i in 1:model.m #Iterate over the points
        Jnm,Jmm = computeIndPointsJ(model,i)
        for j in 1:dim #iterate over the dimensions
            ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
            Jtilde = -sum(ι.*(Kmn.'),2)-sum(model.κ.*Jnm[j,:,:],2)
            V = model.invKmm*Jmm[j,:,:]
            gradients_inducing_points[i,j] = 0.5*(sum((V*model.invKmm-model.StochCoeff*(ι'*Θ*model.κ+model.κ'*Θ*ι)).*transpose(B))-trace(V)-model.StochCoeff*sum(Θ.*Jtilde))
             + model.StochCoeff*0.5*dot(model.y[model.MBIndices],ι*model.μ)
        end
    end
    return gradients_kernel_param,gradients_kernel_coeff,gradients_inducing_points
end


function computefstar(model::FullBatchModel,X_test)
    n = size(X_test,1)
    ksize = model.nSamples
    if model.DownMatrixForPrediction == 0
        if model.TopMatrixForPrediction == 0
            model.TopMatrixForPrediction = model.invK*model.μ
        end
      model.DownMatrixForPrediction = -(model.invK*(eye(ksize)-model.ζ*model.invK))
    end
    kstar = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.X)
    A = kstar*model.invK
    kstarstar = CreateDiagonalKernelMatrix(X_test,model.Kernel_function)
    meanfstar = kstar*model.TopMatrixForPrediction
    covfstar = kstarstar + diag(A*(model.ζ*model.invK-eye(model.nSamples))*transpose(kstar))
    return meanfstar,covfstar
end

function computefstar(model::GibbsSamplerGPC,X_test)
    n = size(X_test,1)
    ksize = model.nSamples
    kstar = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.X)
    A = kstar*model.invK
    kstarstar = CreateDiagonalKernelMatrix(X_test,model.Kernel_function)
    meanfstar = A*model.μ
    covfstar = kstarstar + diag(A*(model.ζ*model.invK-eye(model.nSamples))*transpose(kstar))# + mean(broadcast(x->(meanfstar-A*x).^2,model.estimate))[1]
    # covfstar = kstarstar - diag(kstar*model.invK*transpose(kstar))
    return meanfstar,covfstar
end






# function computeHyperParametersGradients(model::SparseXGPC,iter::Integer)
#     gradients = zeros(model.nKernels)
#     #Compute general values necessary for all gradients (avoid code repetition)
#     B = model.μ*transpose(model.μ) + model.ζ
#     Kmn = CreateKernelMatrix(model.inducingPoints,model.Kernel_function;X2=model.X[model.MBIndices,:])
#     Θ = Diagonal(0.25./model.α[model.MBIndices].*tanh.(0.5*model.α[model.MBIndices]))
#     #If multikernels only update the weight of the kernels, else update the kernel lengthscale
#   for i in 1:model.nKernels
#
#     #ι is the derivative of κ
#     ι = (Jnm-model.κ*Jmm)*model.invKmm
#     Jtilde = Jnn + sum(ι.*(Kmn.'),2)+ sum(model.κ.*Jnm,2)
#     V = model.invKmm*Jmm
#     gradients[i] = 0.5*(sum((V*model.invKmm-model.StochCoeff*(ι'*Θ*model.κ+model.κ'*Θ*ι)).*B)-trace(V)-model.StochCoeff*sum(Θ.*Jtilde))
#      + model.StochCoeff*0.5*dot(model.y[model.MBIndices],ι*model.μ)
#   end
#     elseif model.Kernels[1].Nparams > 0 #Update of the hyperparameters of the KernelMatrix
#         #Compute the derivative of the kernel matrices against the kernel hyperparameter
#       Jnm = model.Kernels[1].coeff*CreateKernelMatrix(model.X[model.MBIndices,:],model.Kernels[1].compute_deriv,X2=model.inducingPoints)
#       Jnn = model.Kernels[1].coeff*CreateDiagonalKernelMatrix(model.X[model.MBIndices,:],model.Kernels[1].compute_deriv)
#       Jmm = model.Kernels[1].coeff*CreateKernelMatrix(model.inducingPoints,model.Kernels[1].compute_deriv)
#       #ι is the derivative of κ
#       ι = (Jnm-model.κ*Jmm)*model.invKmm
#       Jtilde = Jnn + sum(ι.*(Kmn.'),2)+ sum(model.κ.*Jnm,2)
#       V = model.invKmm*Jmm
#       gradients[1] = 0.5*(sum((V*model.invKmm-model.StochCoeff*(ι'*Θ*model.κ+model.κ'*Θ*ι)).*B)-trace(V)-model.StochCoeff*sum(Θ.*Jtilde)+ model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
#       if model.VerboseLevel > 2
#         println("Grad kernel: $(gradients[1]), new param is $(model.Kernels[1].param)")
#       end
#     end
#     return gradients
# end


#### Extra tools for debugging, yet to implement ####


####### TODO #########
function derivativesELBO(model::BatchXGPC)
    return [0,0,0,0]
end

####### TODO #########
function derivativesELBO(model::SparseXGPC)
    return [0,0,0,0]
end
