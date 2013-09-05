cpdef double SampleUniform()
cpdef double SampleGamma(double, double)
cpdef double CalculateDirichletMultinomialLikelihood(long[:], double[:])
cpdef double CalculateGammaLn(double)
cpdef long SampleCategorical(double[:])
cpdef SampleDirichletMultinomial(long[:], double[:])
cpdef double LogSumExp(double, double)
cpdef long SampleCategoricalLn(double[:])
cpdef double SampleNormal(double, double)
cpdef double CalculateMultinomialLogLikelihood(long[:], double[:])
cpdef SetRandomSeed(long)
cpdef double LogSumExp(double, double)
cpdef double LogSumExpArray(double[:])


cdef class normal_parms:
  cdef public double mu
  cdef public double sd


cdef class normal_gamma_parms:
  cdef public double shape
  cdef public double rate
  cdef public double nu
  cdef public double mu  


cpdef normal_parms SampleNormalGamma(normal_gamma_parms)
cpdef normal_gamma_parms NormalGammaPosteriorParms(normal_gamma_parms, double[:])
cpdef normal_parms NormalPosteriorParms(double, double, double, double[:])
cpdef double NormalLike(double, normal_parms)
cpdef double NormalPredictiveLogLikelihood(double, double, double, double, double[:])
