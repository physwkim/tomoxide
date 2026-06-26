#define BS1 32
#define BS2 32
#define BS3 1
#define Pole static_cast<real>(-0.267949192431123f)

#ifdef HALF 
    #define CUDA_R CUDA_R_16F
    #define CUDA_C CUDA_C_16F
    #define mexp(x) hexp(x)
    #define msin(x) hsin(x)
    #define mcos(x) hcos(x)
    typedef half real;
    typedef half2 real2;
    #define TEX2D_L(tex,x,y,z) (__float2half_rn(tex2DLayered<float>(tex,x,y,z)))
    #define CUDA_CREATE_CHANNEL_DESC() cudaCreateChannelDescHalf()
    // surf2DLayeredwrite has no half overload; store the raw 16 bits as a
    // ushort into the 16F array (the texture reads them back as half).
    #define SURF_WRITE_L(val,surf,xb,y,layer) surf2DLayeredwrite(__half_as_ushort(val), surf, xb, y, layer)
#else
    #define CUDA_R CUDA_R_32F
    #define CUDA_C CUDA_C_32F
    #define mexp(x) __expf(x)
    #define msin(x) __sinf(x)
    #define mcos(x) __cosf(x)
    typedef float real;
    typedef float2 real2;        
    #define TEX2D_L(tex,x,y,z) tex2DLayered<float>(tex,x,y,z)
    #define CUDA_CREATE_CHANNEL_DESC() cudaCreateChannelDesc<float>()
    #define SURF_WRITE_L(val,surf,xb,y,layer) surf2DLayeredwrite(val, surf, xb, y, layer)
#endif
