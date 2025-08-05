!====================================================================================
!====================================================================================
! WARM PLASMA MATHEMATICAL MODEL SOLVER
! This code solves a dimensionless warm plasma mathematical model using spectral methods with ifort compiler
!
! GOVERNING EQUATIONS:
! ----> Electron continuity equation:
!       T ∂/∂t ne + (Pe ∇φ + ...) · ∇ne = ∇^2 ne + (...) + GAMA Pe (ni − ne) ne + Da ne exp(-Ea/Te)
!
! ----> Ion continuity equation:
!       T β ∂/∂t ni − Pi ∇φ · ∇ni = ∇^2 ni + β Da ne exp(-Ea/Te) − GAMA Pi (ni − ne) ni
!
! ----> Energy equation:
!       [Energy conservation equation - not shown in header]
!
! ----> Poisson equation for electric potential:
!       ∇^2 φ = GAMA (ne − ni)
!====================================================================================
!====================================================================================

program plasma_cyl
    ! CUDA modules for GPU computation
    use cusolverDn          ! CUDA linear algebra solver
    use cudafor            ! CUDA Fortran interface
    implicit none

!====================================================================================
! VARIABLE DECLARATIONS
!====================================================================================

    !-----------------------------------
    ! GPU-related variables for CUDA solver
    !-----------------------------------
    integer :: t1,t2,ccr,istat,Lwork
    real(8), device, allocatable :: workspace_d(:)              ! GPU workspace
    integer, device :: devInfo_d                                ! Device info for error checking

    ! Pivot arrays for LU decomposition on GPU (for each equation)
    integer, device, dimension(:), allocatable :: devIpiv_i_d   ! Ion equation pivots
    integer, device, dimension(:), allocatable :: devIpiv_e_d   ! Electron equation pivots
    integer, device, dimension(:), allocatable :: devIpiv_w_d   ! Energy equation pivots
    integer, device, dimension(:), allocatable :: devIpiv_P_d   ! Poisson equation pivots

    ! CUDA solver handles
    type(cusolverDnHandle) :: handle,handle2

    !-----------------------------------
    ! General computational variables
    !-----------------------------------
    integer i,i0,i1,i2,i3,j,nx,nxl                            ! Loop indices and grid size
    double precision :: k_delta,ppi,summ,S,R1,R2,CTF,Cr       ! Mathematical constants
    double precision :: r_1,r_2,x1,x2,x3,re1,re2,re3         ! Geometric parameters

    ! Arrays for Gauss-Lobatto grid points and source terms
    double precision, dimension(:),allocatable :: &
               x,r,                    & ! Grid coordinates (x: computational, r: physical)
               fe,fi,fP,fW,fTe,       & ! Source terms for each equation
               he,hi,hP,hW,hTe          ! Modified source terms

    !-----------------------------------
    ! Boundary condition coefficients (Robin BC)
    !-----------------------------------
    ! For electrons (e), ions (i), potential (P), energy (W), and temperature (T)
    ! Each has coefficients at both boundaries (plus/minus)
    double precision :: alfa_plus_e,alfa_minus_e,beta_plus_e,beta_minus_e,g_plus_e,g_minus_e
    double precision :: alfa_plus_i,alfa_minus_i,beta_plus_i,beta_minus_i,g_plus_i,g_minus_i
    double precision :: alfa_plus_P,alfa_minus_P,beta_plus_P,beta_minus_P,g_plus_P,g_minus_P
    double precision :: alfa_plus_W,alfa_minus_W,beta_plus_W,beta_minus_W,g_plus_W,g_minus_W
    double precision :: alfa_plus_T,alfa_minus_T,beta_plus_T,beta_minus_T,g_plus_T,g_minus_T

    ! Additional BC coefficients for Chebyshev implementation
    double precision :: c0_plus_e,c0_minus_e,cN_plus_e,cN_minus_e,e_e
    double precision :: c0_plus_i,c0_minus_i,cN_plus_i,cN_minus_i,e_i
    double precision :: c0_plus_P,c0_minus_P,cN_plus_P,cN_minus_P,e_P
    double precision :: c0_plus_W,c0_minus_W,cN_plus_W,cN_minus_W,e_W
    double precision :: c0_plus_T,c0_minus_T,cN_plus_T,cN_minus_T,e_T

    !-----------------------------------
    ! Chebyshev derivative matrices
    !-----------------------------------
    double precision, dimension(:,:),allocatable :: d1x,d2x    ! General 1st and 2nd derivatives
    double precision, dimension(:,:),allocatable :: b_e,b_i,b_P,b_W,b_T  ! Basis functions

    ! First and second derivative operators for each equation
    double precision, dimension(:,:),allocatable :: DX_e,DXX_e,Dr_e,Drr_e  ! Electron derivatives
    double precision, dimension(:,:),allocatable :: DX_i,DXX_i,Dr_i,Drr_i  ! Ion derivatives
    double precision, dimension(:,:),allocatable :: DX_P,DXX_P,Dr_P,Drr_P  ! Potential derivatives
    double precision, dimension(:,:),allocatable :: DX_W,DXX_W,Dr_W,Drr_W  ! Energy derivatives
    double precision, dimension(:,:),allocatable :: DX_T,DXX_T,Dr_T,Drr_T  ! Temperature derivatives

    !-----------------------------------
    ! Time integration coefficients
    !-----------------------------------
    ! 4th order Adams-Bashforth/Backward Differentiation coefficients
    double precision :: a0,a1,a2,a3,a4,b0,b1,b2,b3

    !-----------------------------------
    ! Solution matrices and arrays
    !-----------------------------------
    ! System matrices for implicit time stepping
    double precision, dimension(:,:),allocatable :: A_matrix_e,A_matrix_inv_e  ! Electron system
    double precision, dimension(:,:),allocatable :: A_matrix_i,A_matrix_inv_i  ! Ion system
    double precision, dimension(:,:),allocatable :: A_matrix_P,A_matrix_inv_P  ! Poisson system
    double precision, dimension(:,:),allocatable :: A_matrix_W,A_matrix_inv_W  ! Energy system
    double precision, dimension(:,:),allocatable :: A_matrix_T,A_matrix_inv_T  ! Temperature system

    ! Time history storage for multi-step methods (n, n-1, n-2, n-3)
    double precision, dimension(:,:),allocatable :: Bne,Bne_1,Bne_2,Bne_3     ! Electron density
    double precision, dimension(:,:),allocatable :: Bni,Bni_1,Bni_2,Bni_3     ! Ion density
    double precision, dimension(:,:),allocatable :: BnW,BnW_1,BnW_2,BnW_3     ! Energy
    double precision, dimension(:,:),allocatable :: BnT,BnT_1,BnT_2,BnT_3     ! Temperature

    ! Additional matrices for time integration
    double precision, dimension(:,:),allocatable :: Cne,Cne_1,Cne_2,Cne_3     ! Electron RHS
    double precision, dimension(:,:),allocatable :: Cni,Cni_1,Cni_2,Cni_3     ! Ion RHS
    double precision, dimension(:,:),allocatable :: CnW,CnW_1,CnW_2,CnW_3     ! Energy RHS
    double precision, dimension(:,:),allocatable :: CnT,CnT_1,CnT_2,CnT_3     ! Temperature RHS

    !-------------------------------------------------------------------
    ! GPU Arrays (device memory)
    !-------------------------------------------------------------------
    ! System matrices on GPU
    double precision, device, dimension(:,:),allocatable :: A_matrix_e_d
    double precision, device, dimension(:,:),allocatable :: A_matrix_i_d
    double precision, device, dimension(:,:),allocatable :: A_matrix_W_d
    double precision, device, dimension(:,:),allocatable :: A_matrix_P_d

    ! Time history matrices on GPU
    double precision, device, dimension(:,:),allocatable :: Bne_d,Bne_1_d,Bne_2_d,Bne_3_d
    double precision, device, dimension(:,:),allocatable :: Bni_d,Bni_1_d,Bni_2_d,Bni_3_d
    double precision, device, dimension(:,:),allocatable :: BnW_d,BnW_1_d,BnW_2_d,BnW_3_d

    double precision, device, dimension(:,:),allocatable :: Cne_d,Cne_1_d,Cne_2_d,Cne_3_d
    double precision, device, dimension(:,:),allocatable :: Cni_d,Cni_1_d,Cni_2_d,Cni_3_d
    double precision, device, dimension(:,:),allocatable :: CnW_d,CnW_1_d,CnW_2_d,CnW_3_d

    ! Source terms on GPU
    double precision, device, dimension(:),allocatable :: hi_d,he_d,hW_d,hP_d

    !-----------------------------------
    ! Solution variables
    !-----------------------------------
    ! Density arrays (current and time history)
    double precision, dimension(:),allocatable :: ne,nen,nen_1,nen_2,nen_3    ! Electron density
    double precision, dimension(:),allocatable :: ni,nin,nin_1,nin_2,nin_3    ! Ion density
    double precision, dimension(:),allocatable :: We,Wen,Wen_1,Wen_2,Wen_3    ! Energy density
    double precision, dimension(:),allocatable :: Te,Ten,P,Pn,En,ME_e         ! Temperature, potential, field

    ! Arrays for storing time evolution
    double precision, dimension(:),allocatable :: r_t,ni_t,ne_t,P_t,We_t,Te_t

    ! Velocity and acceleration arrays
    double precision, dimension(:),allocatable :: a_W,v_W    ! Energy transport
    double precision, dimension(:),allocatable :: a_e,v_e    ! Electron transport
    double precision, dimension(:),allocatable :: a_i,v_i    ! Ion transport

    ! Flux arrays
    double precision, dimension(:),allocatable :: Jen,Jin,qen,R_i             ! Current densities
    double precision, dimension(:),allocatable :: Je_Diff,Ji_Diff             ! Diffusive fluxes
    double precision, dimension(:),allocatable :: Je_Drift,Ji_Drift           ! Drift fluxes

    ! Derivative arrays
    double precision, dimension(:),allocatable :: Dr_nen,Dr_nin,Dr_Wen,Dr_Ten,Dr_Pn    ! First derivatives
    double precision, dimension(:),allocatable :: Drr_nen,Drr_nin,Drr_Wen,Drr_Ten,Drr_Pn ! Second derivatives

!===============================================================================================================
! NOISE-RELATED VARIABLES
!===============================================================================================================
    double precision :: noise_factor                          ! Amplitude of white noise
    double precision, dimension(:),allocatable :: s1,s2       ! Noise arrays
    double precision, dimension(:),allocatable :: Dx_s1,Dx_s2 ! Derivatives of noise

!===============================================================================================================
! ANIMATION AND OUTPUT VARIABLES
!===============================================================================================================
    integer :: nplot                                          ! Iteration interval for plotting
    double precision :: deltatime                             ! Time interval for animation

!===============================================================================================================
! TIME INTEGRATION VARIABLES
!===============================================================================================================
    integer(8) :: nmax                                        ! Total number of time steps
    integer :: K,nit,nprint,kT                              ! Iteration counters
    double precision :: delta_e,delta_i,delta_P,delta_W      ! Convergence parameters
    double precision :: ht,time,t                            ! Time step and total time

!====================================================================================
! PHYSICAL PARAMETERS
!====================================================================================

    !-----------------------------------
    ! DIMENSIONAL PARAMETERS (SI/CGS units)
    !-----------------------------------
    double precision :: &
        nu,           & ! Frequency [1/s]
        tau,          & ! Characteristic time [s]
        me,mi,        & ! Electron/ion mobility [cm^2/V s]
        de,Di,        & ! Electron/ion diffusivity [cm^2/s]
        Ei,           & ! Ionization energy [J]
        Te0,Tec,Ti0,  & ! Reference temperatures [J]
        Hei,          & ! Ionization enthalpy [J]
        ki0,          & ! Ionization rate coefficient [cm^3/s]
        N,            & ! Neutral density [cm^-3]
        n0,           & ! Reference plasma density [cm^-3]
        q0,J0,        & ! Reference current/energy flux
        VRF,VDCRF,    & ! RF voltages [V]
        VDC,V0,       & ! DC and reference voltage [V]
        L,R0,         & ! Chamber dimensions [cm]
        KB,KB1,       & ! Boltzmann constant [J/K]
        e_0,          & ! Elementary charge [C]
        perm_void,    & ! Vacuum permittivity [C/V cm]
        ME_0,         & ! Mean energy
        Pg,Ng,Tg        ! Gas pressure, density, temperature

    double precision :: &
        Pg_bar,       & ! Pressure in bar
        Pg_Nmm2,      & ! Pressure in N/mm^2
        kr,ks           ! Reaction rate constants

    !-----------------------------------
    ! DIMENSIONLESS PARAMETERS
    !-----------------------------------
    double precision :: &
        Da,           & ! Damköhler number (ionization rate)
        Pe,Pi,        & ! Péclet numbers (electron/ion)
        l3,           & ! Length scale parameter
        B,            & ! Beta (diffusivity ratio)
        GAMA,         & ! Gamma (Debye parameter)
        Y,            & ! Secondary emission coefficient
        H,            & ! Enthalpy parameter
        xe,X0,        & ! Energy scale parameters
        D_o,          & ! Diffusion parameter
        Ea,           & ! Activation energy (dimensionless)
        D_r,          & ! Recombination parameter
        K_s             ! Surface reaction parameter

!====================================================================================
! PARAMETER INITIALIZATION
!====================================================================================

    write(*,*)'Dimensional parameters'
    write(*,*)' '

    ! Define chamber geometry
    r_1 = 0.5d0          ! Inner radius [cm]
    r_2 = 1.75d0         ! Outer radius [cm]
    L = r_2-r_1          ! Interelectrode spacing [cm]
    R0= L                ! Reference length scale [cm]

    ! Physical constants
    perm_void = 8.85d-14 ! Vacuum permittivity [C*V^-1*cm^-1]
    e_0= 1.6d-19         ! Elementary charge [C]
    KB = 8.621738d-5*e_0 ! Boltzmann constant [J/K]

    ! Time and frequency scales
    tau= 1.d-9           ! Characteristic time step [s]
    nu = 13.56d6         ! RF frequency [1/s]

    ! Secondary emission
    Y = 0.046d0          ! Secondary electron emission coefficient

    ! Applied voltages
    VDC= 220.d0          ! DC voltage [V]
    V0 = 460.d0          ! Reference voltage [V]

    ! Gas pressure and density calculation
    ! Note: 1 Torr = 1.3332236842 mbar = 1.3332E-4 N/mm^2
    ! Using ideal gas law: P = N k T
    Pg_bar = 0.5d-3      ! Pressure in bar
    Pg_Nmm2 = 0.1d0*Pg_bar ! Convert to N/mm^2
    N = Pg_Nmm2/(273.d0*KB) ! Neutral particle density [cm^-3]

    ! Reference plasma density
    n0 = 4.d9            ! Reference particle density [cm^-3]

    ! Temperature parameters
    Te0= 1.d0*e_0        ! Reference electron temperature [J]
    Tec= 0.5d0*e_0       ! Cathode electron temperature [J]

    ! Reaction energies
    Ei = 24.d0*e_0       ! Ionization activation energy [J]
    Hei= 15.578d0*e_0    ! Ionization enthalpy loss [J]

    ! Reaction rate coefficients
    ki0= 2.5d-7          ! Ionization rate prefactor [cm^3/s]
    kr= 2.5d-7           ! Recombination rate prefactor [cm^3/s]
    ks = 1.19d7          ! Surface reaction rate

    ! Transport coefficients
    Di = 1.d2            ! Ion diffusivity [cm^2/s]
    de = 1.d6            ! Electron diffusivity [cm^2/s]
    me = 2.d5            ! Electron mobility [cm^2/V s]
    mi = 2.d3            ! Ion mobility [cm^2/V s]

    ! Reference scales
    J0 = de*n0/L         ! Characteristic current density
    q0 = J0*Hei          ! Characteristic energy flux
    ME_0= (3.d0/2.d0)*KB*Te0/KB ! Mean energy

    !-----------------------------------
    ! Calculate dimensionless parameters
    !-----------------------------------
    write(*,*)'Dimensionless parameters'
    write(*,*)' '

    ! Transport numbers
    Pe = me*V0/de        ! Electron Péclet number (drift/diffusion)
    Pi = mi*V0/Di        ! Ion Péclet number
    B = de/Di            ! Diffusivity ratio

    ! Time scale parameter
    l3 = 1.d0            ! Simplified (originally: nu*(R0**2.d0)/de)

    ! Electrical parameter
    GAMA = e_0*n0*R0**2.d0/(perm_void*V0) ! Debye parameter

    ! Energy parameters
    D_o = de*e_0/(me*KB*Te0/KB)           ! Einstein relation check
    Ea = Ei/(KB*Te0/KB)                   ! Dimensionless activation energy
    xe = e_0*V0/Hei                       ! Voltage/enthalpy ratio
    H = Hei/((3.d0/2.d0)*KB*Te0/KB)      ! Enthalpy parameter
    X0 = xe*H                             ! Combined energy parameter

    ! Reaction parameters
    Da = (ki0*N*R0**2.d0)/de              ! Damköhler number (ionization)
    D_r= (kr*n0*R0**2.d0)/de              ! Recombination parameter
    K_s = ks*R0/de                        ! Surface reaction parameter

    ! Print dimensionless parameters
    write(*,*)'Pe=',Pe
    write(*,*)'Pi=',Pi
    write(*,*)'Da=',Da
    write(*,*)'B=',B
    write(*,*)'GAMA=',GAMA
    write(*,*)'l3=',l3
    write(*,*)'Y=',Y
    write(*,*)'H=',H
    write(*,*)'Ea=',Ea
    write(*,*)'X0=',X0
    write(*,*)'D_o=',D_o
    write(*,*)'N=',N
    write(*,*)'n0=',n0
    write(*,*)' '

!====================================================================================
! GRID AND TIME SETUP
!====================================================================================

    ! Chebyshev spectral grid setup
    nx = 169             ! Number of Gauss-Lobatto collocation points
    nxl = nx - 1         ! Number of internal points (excluding boundaries)

    ! Geometric center of domain
    i0 = nint(nx/2.d0)   ! Index at domain center

    ! Time marching parameters
    time = 200.d0        ! Total simulation time (dimensionless)
    ht = 1.d-7           ! Time step for numerical stability
    nmax = (time/ht)     ! Total number of time steps

    ! Display simulation parameters
    write(*,*)'Dimensionless time t/t_0=',time
    write(*,*)'Dimensionless step time dt=',ht
    write(*,*)'Total number of time steps nmax=',nmax
    write(*,*)'Total number of nodes nx=',nx
    write(*,*)'Dimensionless Potential at cathode -V_DC/V_0=',-VDC/V0
    write(*,*)'Dimensional Potential at cathode -V_DC=',-VDC
    write(*,*)'Dimensionless Energy at cathode Te_c/Te_0=',Te0/Te0
    write(*,*)' '

    ! Output control
    nprint=1             ! Frequency of screen output

    ! Animation parameters
    deltatime = 1.d0     ! Time interval between animation frames
    nplot = nint(deltatime/ht) ! Iterations per frame

    ! Noise control
    noise_factor = 0.d0  ! Amplitude of white noise (0 = no noise)

!====================================================================================
! TIME INTEGRATION COEFFICIENTS
!====================================================================================

    ! 4th Order Adams-Bashforth/Backward Differentiation coefficients
    ! For implicit time stepping: a0*u^{n+1} = b0*u^n + b1*u^{n-1} + b2*u^{n-2} + b3*u^{n-3}
    a0 = 25.d0/12.d0     ! Coefficient for n+1 term
    a1 = -4.0d0          ! Not used in current implementation
    a2 = 3.0d0           ! Not used in current implementation
    a3 = -4.0d0/3.d0     ! Not used in current implementation
    a4 = 1.0d0/4.d0      ! Not used in current implementation

    b0 = 4.0d0           ! Coefficient for n term
    b1 = -6.0d0          ! Coefficient for n-1 term
    b2 = 4.0d0           ! Coefficient for n-2 term
    b3 = -1.0d0          ! Coefficient for n-3 term

!====================================================================================
! MEMORY ALLOCATION
!====================================================================================

    !-----------------------------------
    ! Grid and source term arrays
    !-----------------------------------
    allocate(x(0:nx),r(0:nx))                    ! Grid coordinates
    allocate(fe(nxl),fi(nxl),fP(nxl))          ! Source terms for continuity/Poisson
    allocate(fW(nxl),fTe(nxl))                  ! Source terms for energy/temperature
    allocate(he(nxl),hi(nxl),hP(nxl))          ! Modified source terms
    allocate(hW(nxl),hTe(nxl))                  ! Modified energy source terms

    !-----------------------------------
    ! Chebyshev basis and derivatives
    !-----------------------------------
    ! Basis function matrices for boundary conditions
    allocate(b_e(0:nx,1:nxl),b_i(0:nx,1:nxl))  ! Electron and ion basis
    allocate(b_P(0:nx,1:nxl),b_W(0:nx,1:nxl))  ! Potential and energy basis
    allocate(b_T(0:nx,1:nxl))                   ! Temperature basis

    ! General Chebyshev derivative matrices
    allocate(d1x(0:nx,0:nx),d2x(0:nx,0:nx))    ! First and second derivatives

    ! Equation-specific derivative operators (internal points only)
    allocate(DX_e(nxl,nxl),DXX_e(nxl,nxl))     ! Electron: d/dx, d²/dx²
    allocate(Dr_e(nxl,nxl),Drr_e(nxl,nxl))     ! Electron: d/dr, (1/r)d/dr + d²/dr²
    allocate(DX_i(nxl,nxl),DXX_i(nxl,nxl))     ! Ion: d/dx, d²/dx²
    allocate(Dr_i(nxl,nxl),Drr_i(nxl,nxl))     ! Ion: d/dr, (1/r)d/dr + d²/dr²
    allocate(DX_P(nxl,nxl),DXX_P(nxl,nxl))     ! Potential: d/dx, d²/dx²
    allocate(Dr_P(nxl,nxl),Drr_P(nxl,nxl))     ! Potential: d/dr, (1/r)d/dr + d²/dr²
    allocate(DX_W(nxl,nxl),DXX_W(nxl,nxl))     ! Energy: d/dx, d²/dx²
    allocate(Dr_W(nxl,nxl),Drr_W(nxl,nxl))     ! Energy: d/dr, (1/r)d/dr + d²/dr²
    allocate(DX_T(nxl,nxl),DXX_T(nxl,nxl))     ! Temperature: d/dx, d²/dx²
    allocate(Dr_T(nxl,nxl),Drr_T(nxl,nxl))     ! Temperature: d/dr, (1/r)d/dr + d²/dr²

    !-----------------------------------
    ! Flux and transport arrays
    !-----------------------------------
    allocate(Jen(0:nx),Jin(0:nx),qen(0:nx))    ! Particle and energy fluxes
    allocate(Je_Diff(0:nx),Ji_Diff(0:nx))      ! Diffusive components
    allocate(Je_Drift(0:nx),Ji_Drift(0:nx))    ! Drift components

    !-----------------------------------
    ! System matrices for implicit solver
    !-----------------------------------
    allocate(A_matrix_e(nxl,nxl),A_matrix_inv_e(nxl,nxl)) ! Electron system
    allocate(A_matrix_i(nxl,nxl),A_matrix_inv_i(nxl,nxl)) ! Ion system
    allocate(A_matrix_P(nxl,nxl),A_matrix_inv_P(nxl,nxl)) ! Poisson system
    allocate(A_matrix_W(nxl,nxl),A_matrix_inv_W(nxl,nxl)) ! Energy system

    !-----------------------------------
    ! Time history matrices for multi-step method
    !-----------------------------------
    ! B matrices store operator*solution at previous time steps
    allocate(Bne(nxl,nxl),Bne_1(nxl,nxl),Bne_2(nxl,nxl),Bne_3(nxl,nxl)) ! Electron
    allocate(Bni(nxl,nxl),Bni_1(nxl,nxl),Bni_2(nxl,nxl),Bni_3(nxl,nxl)) ! Ion
    allocate(BnW(nxl,nxl),BnW_1(nxl,nxl),BnW_2(nxl,nxl),BnW_3(nxl,nxl)) ! Energy

    ! C matrices store RHS terms at previous time steps
    allocate(Cne(nxl,nxl),Cne_1(nxl,nxl),Cne_2(nxl,nxl),Cne_3(nxl,nxl)) ! Electron
    allocate(Cni(nxl,nxl),Cni_1(nxl,nxl),Cni_2(nxl,nxl),Cni_3(nxl,nxl)) ! Ion
    allocate(CnW(nxl,nxl),CnW_1(nxl,nxl),CnW_2(nxl,nxl),CnW_3(nxl,nxl)) ! Energy

    !-----------------------------------
    ! GPU memory allocation
    !-----------------------------------
    ! Pivot arrays for LU decomposition
    allocate(devIpiv_i_d(nxl), devIpiv_e_d(nxl))
    allocate(devIpiv_w_d(nxl), devIpiv_P_d(nxl))

    ! System matrices on GPU
    allocate(A_matrix_e_d(nxl,nxl))
    allocate(A_matrix_i_d(nxl,nxl))
    allocate(A_matrix_w_d(nxl,nxl))
    allocate(A_matrix_P_d(nxl,nxl))

    ! Time history on GPU
    allocate(Bne_d(nxl,nxl),Bne_1_d(nxl,nxl),Bne_2_d(nxl,nxl),Bne_3_d(nxl,nxl))
    allocate(Bni_d(nxl,nxl),Bni_1_d(nxl,nxl),Bni_2_d(nxl,nxl),Bni_3_d(nxl,nxl))
    allocate(BnW_d(nxl,nxl),BnW_1_d(nxl,nxl),BnW_2_d(nxl,nxl),BnW_3_d(nxl,nxl))

    allocate(Cne_d(nxl,nxl),Cne_1_d(nxl,nxl),Cne_2_d(nxl,nxl),Cne_3_d(nxl,nxl))
    allocate(Cni_d(nxl,nxl),Cni_1_d(nxl,nxl),Cni_2_d(nxl,nxl),Cni_3_d(nxl,nxl))
    allocate(CnW_d(nxl,nxl),CnW_1_d(nxl,nxl),CnW_2_d(nxl,nxl),CnW_3_d(nxl,nxl))

    ! Source terms on GPU
    allocate(hi_d(nxl),he_d(nxl),hW_d(nxl),hP_d(nxl))

    !-----------------------------------
    ! Solution arrays
    !-----------------------------------
    ! Current and historical values of primary variables
    allocate(ne(0:nx),nen(0:nx),nen_1(0:nx),nen_2(0:nx),nen_3(0:nx)) ! Electron density
    allocate(ni(0:nx),nin(0:nx),nin_1(0:nx),nin_2(0:nx),nin_3(0:nx)) ! Ion density
    allocate(We(0:nx),Wen(0:nx),Wen_1(0:nx),Wen_2(0:nx),Wen_3(0:nx)) ! Energy density
    allocate(P(0:nx),Pn(0:nx))                                        ! Electric potential
    allocate(Te(0:nx),Ten(0:nx))                                      ! Electron temperature
    allocate(En(0:nx),ME_e(0:nx))                                     ! Electric field, mean energy

    ! Time evolution storage arrays
    allocate(r_t(0:nx),ni_t(0:nx),ne_t(0:nx))
    allocate(P_t(0:nx),We_t(0:nx),Te_t(0:nx))

    ! Transport coefficient arrays
    allocate(a_W(1:nxl),v_W(1:nxl))    ! Energy transport coefficients
    allocate(a_e(1:nxl),v_e(1:nxl))    ! Electron transport coefficients
    allocate(a_i(1:nxl),v_i(1:nxl))    ! Ion transport coefficients

    ! Spatial derivative arrays
    allocate(Dr_nen(0:nx),Dr_nin(0:nx),Dr_Wen(0:nx))    ! First derivatives
    allocate(Dr_Ten(0:nx),Dr_Pn(0:nx))
    allocate(Drr_nen(0:nx),Drr_nin(0:nx),Drr_Wen(0:nx)) ! Second derivatives
    allocate(Drr_Ten(0:nx),Drr_Pn(0:nx))
    allocate(R_i(0:nx))                                   ! Ionization rate array

    !-----------------------------------
    ! Noise arrays
    !-----------------------------------
    allocate(s1(0:nx),s2(0:nx))              ! White noise sources
    allocate(Dx_s1(0:nx),Dx_s2(0:nx))        ! Derivatives of noise

!====================================================================================
! GRID GENERATION - CHEBYSHEV COLLOCATION POINTS
!====================================================================================

    ! Define pi constant
    ppi = dacos(-1.d0)    ! π = 3.14159...

    ! Define computational domain boundaries
    R1 = r_1/r_2          ! Normalized inner radius (left boundary)
    R2 = 1.d0             ! Normalized outer radius (right boundary)

    ! Coordinate transformation factor
    CTF = 2.d0/(R2-R1)    ! Maps [-1,1] to [R1,R2]

    ! Geometry factor (Cr=0: parallel plates, Cr=1: cylindrical)
    Cr = 1.d0             ! Cylindrical geometry

    ! Generate Chebyshev-Gauss-Lobatto collocation points
    ! x(i) = cos(i*π/N) gives points clustered at boundaries
    do i = 0,nx
        x(i) = dcos(dfloat(i)*ppi/dfloat(nx))              ! Computational coordinate [-1,1]
        r(i) = ((R2 - R1)*x(i) + (R2 + R1))/2.d0          ! Physical coordinate [R1,R2]
    end do

    ! Define special radial positions for diagnostics or boundary conditions
    re1 = 0.34d0          ! First special radius
    re2 = 0.57d0          ! Second special radius
    re3 = 0.8d0           ! Third special radius

    ! Convert physical radii to computational coordinates
    x1 = (2.d0*re1 - R2 - R1)/(R2 - R1)
    x2 = (2.d0*re2 - R2 - R1)/(R2 - R1)
    x3 = (2.d0*re3 - R2 - R1)/(R2 - R1)

    ! Find indices corresponding to special radii
    i1 = nint(nx*dacos(x1)/ppi)
    i2 = nint(nx*dacos(x2)/ppi)
    i3 = nint(nx*dacos(x3)/ppi)

!====================================================================================
! INITIALIZE CHEBYSHEV DERIVATIVE MATRICES
!====================================================================================

    ! Calculate first and second derivative matrices
    ! These are the fundamental building blocks for spatial discretization
    call Md1(d1x,x,nx)    ! First derivative operator d/dx
    call Md2(d2x,x,nx)    ! Second derivative operator d²/dx²

!====================================================================================
!====================================================================================
!                                   INITIAL CONDITION SETUP
!====================================================================================

    ! Open output files for time series data at specific radial positions
    open(2,file = 'ni_vs_t.dat')   ! Ion density time evolution
    open(3,file = 'ne_vs_t.dat')   ! Electron density time evolution
    open(11,file = 'We_vs_t.dat')  ! Energy density time evolution
    open(17,file = 'Te_vs_t.dat')  ! Temperature time evolution
    open(12,file = 'P_vs_t.dat')   ! Electric potential time evolution
    open(16,file = 'E_vs_t.dat')   ! Electric field time evolution
    open(13,file = 'Je_vs_t.dat')  ! Electron current density time evolution
    open(14,file = 'Ji_vs_t.dat')  ! Ion current density time evolution
    open(15,file = 'qe_vs_t.dat')  ! Energy flux time evolution
    !open(12,file='K_S1_Dx_s1.dat') ! Random number diagnostics (commented out)

    ! Initialize time step counter and physical time
    nit = 0     ! Iteration counter (can be set to restart value like 40000000)
    t = 0.d0    ! Physical time

    write(*,*)'nit=',nit
    write(*,*)'Press any number if the parameters are right'
    read(*,*)S  ! Wait for user confirmation before starting

    !-----------------------------------
    ! Option to restart from previous simulation
    !-----------------------------------
    ! Uncomment to read initial condition from file:
    ! open(18,file = 'Desktop_10004.dat',status='old')
    ! do i=0,nx
    !     read(18,6) r_t(i),ne_t(i),ni_t(i),P_t(i),We_t(i),Te_t(i)
    ! enddo
    ! close(18)

    !-----------------------------------
    ! Set initial conditions
    !-----------------------------------
    ! Note on grid orientation:
    ! x(0) = -1   corresponds to RIGHT boundary in physical space
    ! x(nx) = +1  corresponds to LEFT boundary in physical space

    ! Initialize uniform mean energy and temperature
    ME_e = 1.d0    ! Mean electron energy (dimensionless)
    Ten = ME_e     ! Electron temperature = mean energy
    Pn = 0.d0      ! Initial potential (will be solved from Poisson equation)

    ! Set initial density profiles (parabolic profile with background)
    DO i = 0,nx
        ! Ion density: parabolic profile centered in domain
        nin(i) = (3d10/n0)*((1.d0 - r(i))**2.d0)*r(i)**2.d0 + (1d10/n0)

        ! Electron density: same as ion density (quasineutral initial condition)
        nen(i) = (3d10/n0)*((1.d0 - r(i))**2.d0)*r(i)**2.d0 + (1d10/n0)

        ! Energy density: product of mean energy and electron density
        Wen(i) = ME_e(i)*nen(i)
    ENDDO

    ! Initialize time history arrays for multi-step method
    ! Need 3 previous time levels for 4th order method
    nen_1 = nen  ! Electron density at n-1
    nen_2 = nen  ! Electron density at n-2
    nen_3 = nen  ! Electron density at n-3

    nin_1 = nin  ! Ion density at n-1
    nin_2 = nin  ! Ion density at n-2
    nin_3 = nin  ! Ion density at n-3

    Wen_1 = Wen  ! Energy density at n-1
    Wen_2 = Wen  ! Energy density at n-2
    Wen_3 = Wen  ! Energy density at n-3

!====================================================================================
!                               MAIN TIME LOOP INITIALIZATION
!====================================================================================

    ! Initialize random number generator for noise
    CALL RANDOM_SEED()

    !-----------------------------
    ! Timer setup for performance monitoring
    !-----------------------------
    CALL SYSTEM_CLOCK( count_rate = ccr )
    call system_clock(t1)

!====================================================================================
!                               START MAIN TIME INTEGRATION LOOP
!====================================================================================
    ! DO K = 20000001,nmax  ! For restart from specific iteration
    DO K = 1,nmax           ! Normal start from beginning

        ! Update iteration counter and physical time
        nit = nit + 1
        t = nit*ht
        write(*,*) nit  ! Print current iteration

        !-----------------------------------
        ! CALCULATE SPATIAL DERIVATIVES
        !-----------------------------------
        ! Electric field from potential gradient: E = -∇φ
        En = -CTF*MATMUL(d1x,Pn)  ! CTF accounts for coordinate transformation

        ! First derivatives of all variables (using chain rule with CTF)
        Dr_nin = CTF*MATMUL(d1x,nin)    ! ∂ni/∂r
        Dr_nen = CTF*MATMUL(d1x,nen)    ! ∂ne/∂r
        Dr_Wen = CTF*MATMUL(d1x,Wen)    ! ∂We/∂r
        Dr_Ten = CTF*MATMUL(d1x,Ten)    ! ∂Te/∂r

        !-----------------------------------
        ! CALCULATE FLUXES
        !-----------------------------------
        ! Energy flux: qe = -(5/3H)[Pe·E·We + (Te/D₀)·∇We]
        qen = -(5.d0/(3.d0*H))*(Pe*En*Wen + (Ten/D_o)*Dr_Wen)

        ! Electron flux: Je = -Pe·E·ne - (Te/D₀)·∇ne
        Jen = -Pe*En*nen - (Ten/D_o)*Dr_nen

        ! Ion flux: Ji = (Pi·ni·E - ∇ni)/β
        Jin = (Pi*nin*En - Dr_nin)/B

        ! Separate drift and diffusion contributions for diagnostics
        Je_Diff = -(Ten/D_o)*Dr_nen     ! Electron diffusion flux
        Ji_Diff = -Dr_nin/B              ! Ion diffusion flux

        Je_Drift = -Pe*En*nen            ! Electron drift flux
        Ji_Drift = Pi*nin*En/B           ! Ion drift flux

!=======================================================================================================
!                               WHITE NOISE GENERATION
!=======================================================================================================
        ! Generate white noise for stochastic effects
        call G_NOISE(nx,s1,s2)

        ! Calculate spatial derivative of noise
        Dx_s1 = MATMUL(d1x,s1)
        Dx_s1 = noise_factor*Dx_s1  ! Scale by noise amplitude

        ! Optional: second noise field
        ! Dx_s2 = MATMUL(d1x,s2)
        ! Dx_s2 = noise_factor*Dx_s2

        ! Diagnostic output for noise
        ! write(12,*) K, s1(1), Dx_s1(1)

!=======================================================================================================
!                               DATA OUTPUT
!=======================================================================================================
        ! Write time series data at selected radial positions (i1, i2, i3)
        IF ((mod(nit,nprint).eq.0).or.(K.eq.1)) THEN
            write(2,4) t, nin(i1), nin(i2), nin(i3)     ! Ion density
            write(3,4) t, nen(i1), nen(i2), nen(i3)     ! Electron density
            write(11,4) t, Wen(i1), Wen(i2), Wen(i3)    ! Energy density
            write(17,4) t, Ten(i1), Ten(i2), Ten(i3)    ! Temperature
            write(12,4) t, Pn(i1), Pn(i2), Pn(i3)       ! Potential
            write(16,4) t, En(i1), En(i2), En(i3)       ! Electric field
            write(13,4) t, Jen(i1), Jen(i2), Jen(i3)    ! Electron flux
            write(14,4) t, Jin(i1), Jin(i2), Jin(i3)    ! Ion flux
            write(15,4) t, qen(i1), qen(i2), qen(i3)    ! Energy flux
        ENDIF

!=======================================================================================================
!                           BOUNDARY CONDITION SETUP
!=======================================================================================================

        !-----------------------------------
        ! RIGHT BOUNDARY (x=+1, r=R2): CATHODE
        !-----------------------------------
        ! Implements secondary electron emission: Je = -γ·Ji

        ! Potential: Dirichlet condition (fixed voltage)
        alfa_plus_P = 1.d0;  beta_plus_P = 0.d0;  g_plus_P = -VDC/V0

        ! Ion flux: Neumann condition (zero gradient)
        alfa_plus_i = 0.d0;  beta_plus_i = CTF;   g_plus_i = 0.d0

        ! Electron flux: Mixed condition with secondary emission
        ! Je = -Pe·En·ne - (Te/D₀)·∇ne = Ks·ne - γ·Ji
        alfa_plus_e = -Pe*En(0)
        beta_plus_e = -CTF*Ten(0)/D_o
        g_plus_e = K_s*nen(0) - Y*Jin(0)  ! Surface reaction + secondary emission

        ! Energy: Dirichlet condition
        alfa_plus_w = 1.d0;  beta_plus_W = 0.d0;  g_plus_W = ME_e(0)*nen(0)

        ! Temperature: Dirichlet condition (fixed at cathode)
        alfa_plus_T = 1.d0;  beta_plus_T = 0.d0;  g_plus_T = 1.d0

        !-----------------------------------
        ! LEFT BOUNDARY (x=-1, r=R1): ANODE
        !-----------------------------------

        ! Potential: Dirichlet condition (grounded)
        alfa_minus_P = 1.d0;  beta_minus_P = 0.d0;  g_minus_P = 0.d0

        ! Ion flux: Neumann condition (zero gradient)
        alfa_minus_i = 0.d0;  beta_minus_i = CTF;   g_minus_i = 0.d0

        ! Electron density: Dirichlet condition (zero)
        alfa_minus_e = 1.d0;  beta_minus_e = 0.d0;  g_minus_e = 0.d0

        ! Energy density: Dirichlet condition (zero)
        alfa_minus_W = 1.d0;  beta_minus_W = 0.d0;  g_minus_W = 0.d0

        ! Temperature: Neumann condition (energy balance)
        alfa_minus_T = 0.d0;  beta_minus_T = (5.d0/3.d0)*CTF;  g_minus_T = -X0*En(nx)

!=====================================================================================
!                            CHEBYSHEV COEFFICIENT CALCULATION
!=====================================================================================
        ! Calculate coefficients for implementing boundary conditions
        ! in the Chebyshev spectral method

        ! Coefficients at x = -1 (left boundary)
        c0_minus_e = -beta_plus_e*d1x(0,nx)
        c0_minus_i = -beta_plus_i*d1x(0,nx)
        c0_minus_P = -beta_plus_P*d1x(0,nx)
        c0_minus_W = -beta_plus_W*d1x(0,nx)
        c0_minus_T = -beta_plus_T*d1x(0,nx)

        c0_plus_e = alfa_minus_e + beta_minus_e*d1x(nx,nx)
        c0_plus_i = alfa_minus_i + beta_minus_i*d1x(nx,nx)
        c0_plus_P = alfa_minus_P + beta_minus_P*d1x(nx,nx)
        c0_plus_W = alfa_minus_W + beta_minus_W*d1x(nx,nx)
        c0_plus_T = alfa_minus_T + beta_minus_T*d1x(nx,nx)

        ! Coefficients at x = +1 (right boundary)
        cN_plus_e = -beta_minus_e*d1x(nx,0)
        cN_plus_i = -beta_minus_i*d1x(nx,0)
        cN_plus_P = -beta_minus_P*d1x(nx,0)
        cN_plus_W = -beta_minus_W*d1x(nx,0)
        cN_plus_T = -beta_minus_T*d1x(nx,0)

        cN_minus_e = alfa_plus_e + beta_plus_e*d1x(0,0)
        cN_minus_i = alfa_plus_i + beta_plus_i*d1x(0,0)
        cN_minus_P = alfa_plus_P + beta_plus_P*d1x(0,0)
        cN_minus_W = alfa_plus_w + beta_plus_W*d1x(0,0)
        cN_minus_T = alfa_plus_T + beta_plus_T*d1x(0,0)

        ! Determinant for boundary condition system
        e_e = c0_plus_e*cN_minus_e - c0_minus_e*cN_plus_e
        e_i = c0_plus_i*cN_minus_i - c0_minus_i*cN_plus_i
        e_P = c0_plus_P*cN_minus_P - c0_minus_P*cN_plus_P
        e_W = c0_plus_W*cN_minus_W - c0_minus_W*cN_plus_W
        e_T = c0_plus_T*cN_minus_T - c0_minus_T*cN_plus_T

        ! Build basis functions for boundary conditions
        do j=1,nxl
            ! Contributions at left boundary (x = -1)
            b_e(0,j) = -c0_plus_e*beta_plus_e*d1x(0,j) - c0_minus_e*beta_minus_e*d1x(nx,j)
            b_i(0,j) = -c0_plus_i*beta_plus_i*d1x(0,j) - c0_minus_i*beta_minus_i*d1x(nx,j)
            b_P(0,j) = -c0_plus_P*beta_plus_P*d1x(0,j) - c0_minus_P*beta_minus_P*d1x(nx,j)
            b_W(0,j) = -c0_plus_W*beta_plus_W*d1x(0,j) - c0_minus_W*beta_minus_W*d1x(nx,j)
            b_T(0,j) = -c0_plus_T*beta_plus_T*d1x(0,j) - c0_minus_T*beta_minus_T*d1x(nx,j)
        enddo

        do j=1,nxl
            ! Contributions at right boundary (x = +1)
            b_e(nx,j) = -cN_minus_e*beta_minus_e*d1x(nx,j) - cN_plus_e*beta_plus_e*d1x(0,j)
            b_i(nx,j) = -cN_minus_i*beta_minus_i*d1x(nx,j) - cN_plus_i*beta_plus_i*d1x(0,j)
            b_P(nx,j) = -cN_minus_P*beta_minus_P*d1x(nx,j) - cN_plus_P*beta_plus_P*d1x(0,j)
            b_W(nx,j) = -cN_minus_W*beta_minus_W*d1x(nx,j) - cN_plus_W*beta_plus_W*d1x(0,j)
            b_T(nx,j) = -cN_minus_T*beta_minus_T*d1x(nx,j) - cN_plus_T*beta_plus_T*d1x(0,j)
        enddo

!=========================================================================================
!                          CHEBYSHEV DERIVATIVE MATRICES WITH BC
!=========================================================================================
        ! Construct derivative operators that incorporate boundary conditions

        DO i=1,nxl
            DO j=1,nxl
                ! First derivative operators with BC
                DX_e(i,j) = d1x(i,j) + (1.d0/e_e)*(b_e(0,j)*d1x(i,0) + b_e(nx,j)*d1x(i,nx))
                DX_i(i,j) = d1x(i,j) + (1.d0/e_i)*(b_i(0,j)*d1x(i,0) + b_i(nx,j)*d1x(i,nx))
                DX_P(i,j) = d1x(i,j) + (1.d0/e_P)*(b_P(0,j)*d1x(i,0) + b_P(nx,j)*d1x(i,nx))
                DX_W(i,j) = d1x(i,j) + (1.d0/e_W)*(b_W(0,j)*d1x(i,0) + b_W(nx,j)*d1x(i,nx))
                DX_T(i,j) = d1x(i,j) + (1.d0/e_T)*(b_T(0,j)*d1x(i,0) + b_T(nx,j)*d1x(i,nx))

                ! Second derivative operators with BC
                DXX_e(i,j) = d2x(i,j) + (1.d0/e_e)*(b_e(0,j)*d2x(i,0) + b_e(nx,j)*d2x(i,nx))
                DXX_i(i,j) = d2x(i,j) + (1.d0/e_i)*(b_i(0,j)*d2x(i,0) + b_i(nx,j)*d2x(i,nx))
                DXX_P(i,j) = d2x(i,j) + (1.d0/e_P)*(b_P(0,j)*d2x(i,0) + b_P(nx,j)*d2x(i,nx))
                DXX_W(i,j) = d2x(i,j) + (1.d0/e_W)*(b_W(0,j)*d2x(i,0) + b_W(nx,j)*d2x(i,nx))
                DXX_T(i,j) = d2x(i,j) + (1.d0/e_T)*(b_T(0,j)*d2x(i,0) + b_T(nx,j)*d2x(i,nx))

                !-----------------------------------
                ! Apply coordinate transformation
                !-----------------------------------
                ! Transform to physical coordinates including cylindrical geometry term
                ! ∇²u = ∂²u/∂r² + (1/r)∂u/∂r (cylindrical) or just ∂²u/∂r² (planar)

                Drr_e(i,j) = (CTF**2.d0)*DXX_e(i,j) + CTF*(Cr/r(i))*DX_e(i,j)
                Drr_i(i,j) = (CTF**2.d0)*DXX_i(i,j) + CTF*(Cr/r(i))*DX_i(i,j)
                Drr_P(i,j) = (CTF**2.d0)*DXX_P(i,j) + CTF*(Cr/r(i))*DX_P(i,j)
                Drr_W(i,j) = (CTF**2.d0)*DXX_W(i,j) + CTF*(Cr/r(i))*DX_W(i,j)
                Drr_T(i,j) = (CTF**2.d0)*DXX_T(i,j) + CTF*(Cr/r(i))*DX_T(i,j)

                ! First derivatives in physical coordinates
                Dr_e(i,j) = CTF*Dx_e(i,j)
                Dr_i(i,j) = CTF*Dx_i(i,j)
                Dr_P(i,j) = CTF*Dx_P(i,j)
                Dr_W(i,j) = CTF*DX_W(i,j)
                Dr_T(i,j) = CTF*DX_T(i,j)
            ENDDO
        ENDDO

!====================================================================================
!           BUILD SYSTEM MATRICES FOR IMPLICIT TIME STEPPING
!====================================================================================

        do i=1,nxl
            do j=1,nxl
                ! Kronecker delta
                if (i.eq.j) then
                    k_delta = 1.d0
                else
                    k_delta = 0.d0
                end if

                !-----------------------------------
                ! ION EQUATION MATRICES
                !-----------------------------------
                ! Transport coefficients
                v_i(i) = 1.d0                    ! Diffusion coefficient (normalized)
                a_i(i) = -Pi*En(i)               ! Drift coefficient

                ! System matrix: β·l₃·a₀·I - ht·v·∇²
                A_matrix_i(i,j) = B*l3*a0*k_delta - ht*v_i(i)*Drr_i(i,j)

                ! Time history matrices for multi-step method
                Bni(i,j)   = B*l3*a1*k_delta - ht*b0*a_i(i)*Dr_i(i,j)
                Bni_1(i,j) = B*l3*a2*k_delta - ht*b1*a_i(i)*Dr_i(i,j)
                Bni_2(i,j) = B*l3*a3*k_delta - ht*b2*a_i(i)*Dr_i(i,j)
                Bni_3(i,j) = B*l3*a4*k_delta - ht*b3*a_i(i)*Dr_i(i,j)

                !-----------------------------------
                ! ELECTRON EQUATION MATRICES
                !-----------------------------------
                ! Transport coefficients
                v_e(i) = Ten(i)/D_o                           ! Diffusion coefficient
                a_e(i) = Pe*En(i) + (1.d0/D_o)*Dr_Ten(i)     ! Drift + thermophoretic

                ! System matrix
                A_matrix_e(i,j) = l3*a0*k_delta - ht*v_e(i)*Drr_e(i,j)

                ! Time history matrices
                Bne(i,j)   = l3*a1*k_delta - ht*b0*a_e(i)*Dr_e(i,j)
                Bne_1(i,j) = l3*a2*k_delta - ht*b1*a_e(i)*Dr_e(i,j)
                Bne_2(i,j) = l3*a3*k_delta - ht*b2*a_e(i)*Dr_e(i,j)
                Bne_3(i,j) = l3*a4*k_delta - ht*b3*a_e(i)*Dr_e(i,j)

                !-----------------------------------
                ! ENERGY EQUATION MATRICES
                !-----------------------------------
                ! Transport coefficients (5/3 factor from kinetic theory)
                v_W(i) = (5.d0/3.d0)*(Ten(i)/D_o)
                a_W(i) = (5.d0/3.d0)*(Pe*En(i) + (1.d0/D_o)*Dr_Ten(i))

                ! System matrix
                A_matrix_W(i,j) = l3*a0*k_delta - ht*v_W(i)*Drr_W(i,j)

                ! Time history matrices
                BnW(i,j)   = l3*a1*k_delta - ht*b0*a_W(i)*Dr_W(i,j)
                BnW_1(i,j) = l3*a2*k_delta - ht*b1*a_W(i)*Dr_W(i,j)
                BnW_2(i,j) = l3*a3*k_delta - ht*b2*a_W(i)*Dr_W(i,j)
                BnW_3(i,j) = l3*a4*k_delta - ht*b3*a_W(i)*Dr_W(i,j)

                !-----------------------------------
                ! POISSON EQUATION MATRIX
                !-----------------------------------
                ! No time derivative, just Laplacian operator
                A_matrix_P(i,j) = Drr_P(i,j)
            enddo
        enddo

!----------------------------------------------------------------------------
!                            GPU SOLVER SETUP (CUSOLVER)
!----------------------------------------------------------------------------

        ! Copy matrices to GPU memory
        A_matrix_e_d = A_matrix_e
        A_matrix_i_d = A_matrix_i
        A_matrix_w_d = A_matrix_w
        A_matrix_P_d = A_matrix_P

        ! Create cuSOLVER handle
        istat = cusolverDnCreate(handle)

        ! Calculate workspace size needed for LU factorization
        istat = cusolverDnDgetrf_bufferSize(handle,nxl,nxl,A_matrix_e_d,nxl,Lwork)
        istat = cusolverDnDgetrf_bufferSize(handle,nxl,nxl,A_matrix_i_d,nxl,Lwork)
        istat = cusolverDnDgetrf_bufferSize(handle,nxl,nxl,A_matrix_w_d,nxl,Lwork)
        istat = cusolverDnDgetrf_bufferSize(handle,nxl,nxl,A_matrix_P_d,nxl,Lwork)

        ! Allocate workspace on GPU
        allocate(workspace_d(Lwork))

        !-----------------------------------
        ! LU FACTORIZATION OF SYSTEM MATRICES
        !-----------------------------------
        ! Factorize A = L*U for each equation system
        istat = cusolverDnDgetrf(handle, nxl, nxl, A_matrix_e_d, nxl, workspace_d, devIpiv_e_d, devInfo_d)
        istat = cusolverDnDgetrf(handle, nxl, nxl, A_matrix_i_d, nxl, workspace_d, devIpiv_i_d, devInfo_d)
        istat = cusolverDnDgetrf(handle, nxl, nxl, A_matrix_w_d, nxl, workspace_d, devIpiv_w_d, devInfo_d)
        istat = cusolverDnDgetrf(handle, nxl, nxl, A_matrix_P_d, nxl, workspace_d, devIpiv_P_d, devInfo_d)

        !-----------------------------------
        ! SOLVE LINEAR SYSTEMS: C = A⁻¹ × B
        !-----------------------------------
        ! For each equation and time level, solve A·C = B

        ! Current time level (n)
        ! Electrons: Cne = A_e⁻¹ × Bne
        Bne_d = Bne
        istat = cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_e_d, nxl, devIpiv_e_d, Bne_d, nxl, devInfo_d)
        Cne = Bne_d

        ! Ions: Cni = A_i⁻¹ × Bni
        Bni_d = Bni
        istat = cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_i_d, nxl, devIpiv_i_d, Bni_d, nxl, devInfo_d)
        Cni = Bni_d

        ! Energy: CnW = A_W⁻¹ × BnW
        BnW_d = BnW
        istat = cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_w_d, nxl, devIpiv_w_d, BnW_d, nxl, devInfo_d)
        CnW = BnW_d

        ! Time level n-1
        ! Electrons
      !  C^(n-1) = A^-1 x B^(n-1):
        Bne_1_d=Bne_1
        istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_e_d, nxl, devIpiv_e_d, Bne_1_d, nxl, devInfo_d)
        Cne_1=Bne_1_d

        Bni_1_d=Bni_1
        istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_i_d, nxl, devIpiv_i_d, Bni_1_d, nxl, devInfo_d)
        Cni_1=Bni_1_d

        BnW_1_d=BnW_1
        istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_w_d, nxl, devIpiv_w_d, BnW_1_d, nxl, devInfo_d)
        CnW_1=BnW_1_d

        ! C^(n-2) = A^-1 x B^(n-2):
        Bne_2_d=Bne_2
        istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_e_d, nxl, devIpiv_e_d, Bne_2_d, nxl, devInfo_d)
        Cne_2=Bne_2_d

        Bni_2_d=Bni_2
        istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_i_d, nxl, devIpiv_i_d, Bni_2_d, nxl, devInfo_d)
        Cni_2=Bni_2_d

        BnW_2_d=BnW_2
        istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_w_d, nxl, devIpiv_w_d, BnW_2_d, nxl, devInfo_d)
        CnW_2=BnW_2_d

        ! E^(n-3) = A^-1 x B^(n-3):

        Bne_3_d=Bne_3
        istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_e_d, nxl, devIpiv_e_d, Bne_3_d, nxl, devInfo_d)
        Cne_3=Bne_3_d

        Bni_3_d=Bni_3
        istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_i_d, nxl, devIpiv_i_d, Bni_3_d, nxl, devInfo_d)
        Cni_3=Bni_3_d

        BnW_3_d=BnW_3
        istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_w_d, nxl, devIpiv_w_d, BnW_3_d, nxl, devInfo_d)
        CnW_3=BnW_3_d

        deallocate(Workspace_d)
        istat=cusolverDnDestroy(handle)
!----------------------------------------------------------------------------------------
!=======================================================
!                         SOURCE TERMS CALCULATION
!                    FOR ION, ELECTRON, ENERGY AND POTENTIAL EQUATIONS
!=======================================================

    DO i=1,nxl
        !-----------------------------------
        ! ION SOURCE TERM
        !-----------------------------------
        ! fi = -Γ·Pi·(ni-ne)·ni + β·Da·ne·exp(-Ea/Te) + β·∇(noise) - β·Dr·ne·ni
        ! Terms:
        ! -Γ·Pi·(ni-ne)·ni: Drift due to charge separation
        ! β·Da·ne·exp(-Ea/Te): Ionization source
        ! β·∇(noise): Stochastic noise term
        ! -β·Dr·ne·ni: Recombination loss
        fi(i) = -GAMA*Pi*(nin(i) - nen(i))*nin(i) + B*Da*nen(i)*DEXP(-Ea/Ten(i)) + B*Dx_s1(i)- B*D_r*nen(i)*nin(i)

        ! Transport coefficients for ions
        v_i(i) = 1.d0              ! Normalized diffusion coefficient
        a_i(i) = -Pi*En(i)         ! Drift velocity (mobility × E-field)

        ! Modified source term including boundary contributions
        ! hi = ht·fi + boundary correction terms from Robin BC
        hi(i) = ht*fi(i) &
             + v_i(i)*(ht/e_i)*(c0_minus_i*g_minus_i + c0_plus_i*g_plus_i)*(CTF**2.d0)*d2x(i,0) &
             + v_i(i)*(ht/e_i)*(cN_minus_i*g_minus_i + cN_plus_i*g_plus_i)*(CTF**2.d0)*d2x(i,nx) &
             + v_i(i)*(ht/e_i)*(c0_minus_i*g_minus_i + c0_plus_i*g_plus_i)*CTF*(Cr/r(i))*d1x(i,0) &
             + v_i(i)*(ht/e_i)*(cN_minus_i*g_minus_i + cN_plus_i*g_plus_i)*CTF*(Cr/r(i))*d1x(i,nx) &
             + (b0+b1+b2+b3)*a_i(i)*(ht/e_i)*(c0_minus_i*g_minus_i + c0_plus_i*g_plus_i)*CTF*d1x(i,0) &
             + (b0+b1+b2+b3)*a_i(i)*(ht/e_i)*(cN_minus_i*g_minus_i + cN_plus_i*g_plus_i)*CTF*d1x(i,nx)

        !-----------------------------------
        ! ELECTRON SOURCE TERM
        !-----------------------------------
        ! fe = Γ·Pe·(ni-ne)·ne + Da·ne·exp(-Ea/Te) + ∇(noise) - Dr·ne·ni
        ! Similar to ion equation but with opposite sign for drift term
        fe(i) = GAMA*Pe*(nin(i) - nen(i))*nen(i) + Da*nen(i)*DEXP(-Ea/Ten(i)) + Dx_s1(i)- D_r*nen(i)*nin(i)

        ! Transport coefficients for electrons
        v_e(i) = Ten(i)/D_o                           ! Temperature-dependent diffusion
        a_e(i) = Pe*En(i) + (1.d0/D_o)*Dr_Ten(i)     ! Drift + thermophoresis

        ! Modified source term with boundary corrections
        he(i) = ht*fe(i) &
               + v_e(i)*(ht/e_e)*(c0_minus_e*g_minus_e + c0_plus_e*g_plus_e)*(CTF**2.d0)*d2x(i,0) &
               + v_e(i)*(ht/e_e)*(cN_minus_e*g_minus_e + cN_plus_e*g_plus_e)*(CTF**2.d0)*d2x(i,nx) &
               + v_e(i)*(ht/e_e)*(c0_minus_e*g_minus_e + c0_plus_e*g_plus_e)*CTF*(Cr/r(i))*d1x(i,0) &
               + v_e(i)*(ht/e_e)*(cN_minus_e*g_minus_e + cN_plus_e*g_plus_e)*CTF*(Cr/r(i))*d1x(i,nx) &
               + (b0+b1+b2+b3)*a_e(i)*(ht/e_e)*(c0_minus_e*g_minus_e + c0_plus_e*g_plus_e)*CTF*d1x(i,0) &
               + (b0+b1+b2+b3)*a_e(i)*(ht/e_e)*(cN_minus_e*g_minus_e + cN_plus_e*g_plus_e)*CTF*d1x(i,nx)

        !-----------------------------------
        ! ENERGY SOURCE TERM
        !-----------------------------------
        ! fW = (5/3)·Γ·Pe·(ni-ne)·We - X0·Je·E - H·Da·ne·exp(-Ea/Te) + H·Dr·ne·ni
        ! Terms:
        ! (5/3)·Γ·Pe·(ni-ne)·We: Energy drift due to charge separation
        ! -X0·Je·E: Joule heating
        ! -H·Da·ne·exp(-Ea/Te): Energy loss due to ionization
        ! H·Dr·ne·ni: Energy gain from recombination
        fW(i) = (5.d0/3.d0)*GAMA*Pe*(nin(i) - nen(i))*Wen(i) - X0*Jen(i)*En(i)&
              - H*Da*nen(i)*DEXP(-Ea/Ten(i)) + H*D_r*nen(i)*nin(i)

        ! Transport coefficients for energy (5/3 factor from kinetic theory)
        v_W(i) = (5.d0/3.d0)*(Ten(i)/D_o)
        a_W(i) = (5.d0/3.d0)*(Pe*En(i) + (1.d0/D_o)*Dr_Ten(i))

        ! Modified source term with boundary corrections
        hW(i) = ht*fW(i) &
              + v_W(i)*(ht/e_W)*(c0_minus_W*g_minus_W + c0_plus_W*g_plus_W)*(CTF**2.d0)*d2x(i,0) &
              + v_W(i)*(ht/e_W)*(cN_minus_W*g_minus_W + cN_plus_W*g_plus_W)*(CTF**2.d0)*d2x(i,nx) &
              + v_W(i)*(ht/e_W)*(c0_minus_W*g_minus_W + c0_plus_W*g_plus_W)*CTF*(Cr/r(i))*d1x(i,0) &
              + v_W(i)*(ht/e_W)*(cN_minus_W*g_minus_W + cN_plus_W*g_plus_W)*CTF*(Cr/r(i))*d1x(i,nx) &
              + (b0+b1+b2+b3)*a_W(i)*(ht/e_W)*(c0_minus_W*g_minus_W + c0_plus_W*g_plus_W)*CTF*d1x(i,0) &
              + (b0+b1+b2+b3)*a_W(i)*(ht/e_W)*(cN_minus_W*g_minus_W + cN_plus_W*g_plus_W)*CTF*d1x(i,nx)

        !-----------------------------------
        ! POISSON SOURCE TERM
        !-----------------------------------
        ! fP = Γ·(ne - ni): Charge density source
        fP(i) = GAMA*(nen(i) - nin(i))

        ! Modified source term for Poisson equation (steady-state, no time derivative)
        hP(i) = fP(i) - (1.d0/e_P)*(c0_minus_P*g_minus_P + c0_plus_P*g_plus_P)*(CTF**2.d0)*d2x(i,0) &
                      - (1.d0/e_P)*(cN_minus_P*g_minus_P + cN_plus_P*g_plus_P)*(CTF**2.d0)*d2x(i,nx) &
                      - (1.d0/e_P)*(c0_minus_P*g_minus_P + c0_plus_P*g_plus_P)*CTF*(Cr/r(i))*d1x(i,0) &
                      - (1.d0/e_P)*(cN_minus_P*g_minus_P + cN_plus_P*g_plus_P)*CTF*(Cr/r(i))*d1x(i,nx)
    ENDDO

!========================================================================================
!                     SOLVE LINEAR SYSTEMS FOR NEW VARIABLES
!                     ne^(n+1), ni^(n+1), We^(n+1), P^(n+1)
!========================================================================================

    ! Create new cuSOLVER handle for solving the linear systems
    istat = cusolverDnCreate(handle)

    !-----------------------------------
    ! SOLVE FOR ION DENSITY
    !-----------------------------------
    ! Solve: A_matrix_i × ni^(n+1) = hi
    hi_d = hi(1:nxl)    ! Copy to GPU
    istat = cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, 1, A_matrix_i_d, nxl, devIpiv_i_d, hi_d, nxl, devInfo_d)
    ni(1:nxl) = hi_d    ! Copy back from GPU

    !-----------------------------------
    ! SOLVE FOR ELECTRON DENSITY
    !-----------------------------------
    ! Solve: A_matrix_e × ne^(n+1) = he
    he_d = he(1:nxl)
    istat = cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, 1, A_matrix_e_d, nxl, devIpiv_e_d, he_d, nxl, devInfo_d)
    ne(1:nxl) = he_d

    !-----------------------------------
    ! SOLVE FOR ENERGY DENSITY
    !-----------------------------------
    ! Solve: A_matrix_W × We^(n+1) = hW
    hW_d = hW(1:nxl)
    istat = cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, 1, A_matrix_W_d, nxl, devIpiv_w_d, hW_d, nxl, devInfo_d)
    We(1:nxl) = hW_d

    !-----------------------------------
    ! SOLVE FOR ELECTRIC POTENTIAL
    !-----------------------------------
    ! Solve: A_matrix_P × P^(n+1) = hP
    hP_d = hP(1:nxl)
    istat = cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, 1, A_matrix_P_d, nxl, devIpiv_P_d, hP_d, nxl, devInfo_d)
    P(1:nxl) = hP_d

    ! Destroy cuSOLVER handle
    istat = cusolverDnDestroy(handle)

    !-----------------------------------
    ! APPLY TIME STEPPING CORRECTION
    !-----------------------------------
    ! Complete the 4th order backward differentiation formula
    ! u^(n+1) = u^(n+1) - sum(C_k × u^(n-k))

    ! Ion density correction
    ni(1:nxl) = ni(1:nxl) &
              - MATMUL(Cni,nin(1:nxl)) &      ! C^n × ni^n
              - MATMUL(Cni_1,nin_1(1:nxl)) &  ! C^(n-1) × ni^(n-1)
              - MATMUL(Cni_2,nin_2(1:nxl)) &  ! C^(n-2) × ni^(n-2)
              - MATMUL(Cni_3,nin_3(1:nxl))    ! C^(n-3) × ni^(n-3)

    ! Electron density correction
    ne(1:nxl) = ne(1:nxl) &
              - MATMUL(Cne,nen(1:nxl)) &
              - MATMUL(Cne_1,nen_1(1:nxl)) &
              - MATMUL(Cne_2,nen_2(1:nxl)) &
              - MATMUL(Cne_3,nen_3(1:nxl))

    ! Energy density correction
    We(1:nxl) = We(1:nxl) &
              - MATMUL(CnW,Wen(1:nxl)) &
              - MATMUL(CnW_1,Wen_1(1:nxl)) &
              - MATMUL(CnW_2,Wen_2(1:nxl)) &
              - MATMUL(CnW_3,Wen_3(1:nxl))

    !-----------------------------------
    ! CALCULATE TEMPERATURE FROM ENERGY
    !-----------------------------------
    ! Temperature = Mean energy = We/ne
    ME_e(1:nxl) = Wen(1:nxl)/nen(1:nxl)
    Te(1:nxl) = ME_e(1:nxl)

!===================================================================================
!               BOUNDARY VALUE CALCULATIONS FOR ne, ni, P, We
!===================================================================================
    ! Use the basis functions to reconstruct boundary values from interior solution

    !-----------------------------------
    ! ION DENSITY BOUNDARIES
    !-----------------------------------
    ! Right boundary (x=0, r=R2, cathode)
    summ = 0.d0
    do j=1,nxl
        summ = summ + b_i(0,j)*ni(j)
    enddo
    ni(0) = (1.d0/e_i)*summ + (1.d0/e_i)*(c0_minus_i*g_minus_i + c0_plus_i*g_plus_i)

    ! Left boundary (x=nx, r=R1, anode)
    summ = 0.d0
    do j=1,nxl
        summ = summ + b_i(nx,j)*ni(j)
    enddo
    ni(nx) = (1.d0/e_i)*summ + (1.d0/e_i)*(cN_minus_i*g_minus_i + cN_plus_i*g_plus_i)

    !-----------------------------------
    ! ELECTRON DENSITY BOUNDARIES
    !-----------------------------------
    ! Right boundary (cathode)
    summ = 0.d0
    do j=1,nxl
        summ = summ + b_e(0,j)*ne(j)
    enddo
    ne(0) = (1.d0/e_e)*summ + (1.d0/e_e)*(c0_minus_e*g_minus_e + c0_plus_e*g_plus_e)

    ! Left boundary (anode)
    summ = 0.d0
    do j=1,nxl
        summ = summ + b_e(nx,j)*ne(j)
    enddo
    ne(nx) = (1.d0/e_e)*summ + (1.d0/e_e)*(cN_minus_e*g_minus_e + cN_plus_e*g_plus_e)

    !-----------------------------------
    ! ENERGY DENSITY BOUNDARIES
    !-----------------------------------
    ! Right boundary (cathode)
    summ = 0.d0
    do j=1,nxl
        summ = summ + b_W(0,j)*We(j)
    enddo
    We(0) = (1.d0/e_W)*summ + (1.d0/e_W)*(c0_minus_W*g_minus_W + c0_plus_W*g_plus_W)

    ! Left boundary (anode)
    summ = 0.d0
    do j=1,nxl
        summ = summ + b_W(nx,j)*We(j)
    enddo
    We(nx) = (1.d0/e_W)*summ + (1.d0/e_W)*(cN_minus_W*g_minus_W + cN_plus_W*g_plus_W)

    !-----------------------------------
    ! TEMPERATURE BOUNDARIES
    !-----------------------------------
    ! Right boundary (cathode)
    summ = 0.d0
    do j=1,nxl
        summ = summ + b_T(0,j)*Te(j)
    enddo
    Te(0) = (1.d0/e_T)*summ + (1.d0/e_T)*(c0_minus_T*g_minus_T + c0_plus_T*g_plus_T)

    ! Left boundary (anode)
    summ = 0.d0
    do j=1,nxl
        summ = summ + b_T(nx,j)*Te(j)
    enddo
    Te(nx) = (1.d0/e_T)*summ + (1.d0/e_T)*(cN_minus_T*g_minus_T + cN_plus_T*g_plus_T)

    !-----------------------------------
    ! POTENTIAL BOUNDARIES
    !-----------------------------------
    ! Right boundary (cathode)
    summ = 0.d0
    do j=1,nxl
        summ = summ + b_P(0,j)*P(j)
    enddo
    P(0) = (1.d0/e_P)*summ + (1.d0/e_P)*(c0_minus_P*g_minus_P + c0_plus_P*g_plus_P)

    ! Left boundary (anode)
    summ = 0.d0
    do j=1,nxl
        summ = summ + b_P(nx,j)*P(j)
    enddo
    P(nx) = (1.d0/e_P)*summ + (1.d0/e_P)*(cN_minus_P*g_minus_P + cN_plus_P*g_plus_P)

!====================================================================================
! CONVERGENCE CHECK AND DIAGNOSTICS
!====================================================================================
    ! Calculate maximum change in electron density (convergence metric)
    delta_e = MAXVAL(ABS(ne(1:nxl)-nen(1:nxl)))

    ! Print convergence information at specified intervals
    IF(mod(nit,nprint).eq.0) print 1,nit,delta_e,ne(i0),ni(i0),P(i0),We(i0)

!====================================================================================
! UPDATE SOLUTION ARRAYS FOR NEXT TIME STEP
!====================================================================================
    ! Shift time history for 4th order method
    ! n-3 <- n-2 <- n-1 <- n <- new

    ! Electron density history
    nen_3 = nen_2
    nen_2 = nen_1
    nen_1 = nen
    nen = ne

    ! Ion density history
    nin_3 = nin_2
    nin_2 = nin_1
    nin_1 = nin
    nin = ni

    ! Energy density history
    Wen_3 = Wen_2
    Wen_2 = Wen_1
    Wen_1 = Wen
    Wen = We

    ! Update current values
    Pn = P      ! Potential
    Ten = Te    ! Temperature
    ME_e = Ten  ! Mean energy

    ! Calculate ionization rate for diagnostics
    R_i = Da*nen*DEXP(-Ea/Ten)

!====================================================================================
! ANIMATION OUTPUT
!====================================================================================
    ! Save fields for animation at specified intervals
    IF(mod(nit,nplot).eq.0)THEN
        call ANIMATION(nit,nplot,nx,nen,nin,Pn,Wen,Ten,r)
    END IF

!====================================================================================
! REAL-TIME DATA OUTPUT FOR MONITORING
!====================================================================================
    ! Output ionization rate profile
    open(10,file='READ_REAL_TIME_Da_ne.dat')
    do i=0,nx
        write(10,3) r(i),R_i(i),H*R_i(i)
    enddo
    close(10)

    ! Output all field profiles
    open(10,file='READ_REAL_TIME_ni_ne_P_We_Te_E.dat')
    do i=0,nx
        write(10,7) r(i),nin(i),nen(i),Pn(i),Wen(i),Ten(i),En(i)
    enddo
    close(10)

! Format statements for output
1    format(1x,'N=',I8,'  Del=',E10.3,'  ne=',F12.4,'  ni=',F12.4,'  P=',F12.4,'  We=',F12.4)
3    format(1x,E12.5,2x,E12.5,2x,E12.5)
4    format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)
5    format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)
6    format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)
7    format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)

!=================================================================================================================
!=================================================================================================================
END DO    ! <--- END OF MAIN TIME LOOP

!====================================================================================
! EXECUTION TIME MEASUREMENT
!====================================================================================
   !-----------------------------
   ! Stop timer and calculate total execution time
   !-----------------------------
   CALL SYSTEM_CLOCK(t2)
   PRINT*, 'Time by loop (seconds):', (t2-t1)/float(ccr)

!====================================================================================
! CLOSE OUTPUT FILES
!====================================================================================
   ! Close all time series data files
   close(2)   ! ni_vs_t.dat - Ion density at fixed points
   close(3)   ! ne_vs_t.dat - Electron density at fixed points
   close(11)  ! We_vs_t.dat - Energy density at fixed points
   close(17)  ! Te_vs_t.dat - Temperature at fixed points
   close(12)  ! P_vs_t.dat - Electric potential at fixed points
   close(16)  ! E_vs_t.dat - Electric field at fixed points
   close(13)  ! Je_vs_t.dat - Electron current density
   close(14)  ! Ji_vs_t.dat - Ion current density
   close(15)  ! qe_vs_t.dat - Energy flux

!========================================================================================================================
!                        FINAL BOUNDARY VALUE CALCULATIONS
!========================================================================================================================
   ! After completing the simulation, recalculate final values
   ! at boundaries using Chebyshev basis functions

   !-----------------------------------
   ! ION DENSITY AT BOUNDARIES
   !-----------------------------------
   ! Right boundary (x=0, cathode)
   summ = 0.d0
   do j=1,nxl
       summ = summ + b_i(0,j)*nin(j)
   enddo
   nin(0) = (1.d0/e_i)*summ + (1.d0/e_i)*(c0_minus_i*g_minus_i + c0_plus_i*g_plus_i)

   ! Left boundary (x=nx, anode)
   summ = 0.d0
   do j=1,nxl
       summ = summ + b_i(nx,j)*nin(j)
   enddo
   nin(nx) = (1.d0/e_i)*summ + (1.d0/e_i)*(cN_minus_i*g_minus_i + cN_plus_i*g_plus_i)

   !-----------------------------------
   ! ELECTRON DENSITY AT BOUNDARIES
   !-----------------------------------
   ! Right boundary (cathode)
   summ = 0.d0
   do j=1,nxl
       summ = summ + b_e(0,j)*nen(j)
   enddo
   nen(0) = (1.d0/e_e)*summ + (1.d0/e_e)*(c0_minus_e*g_minus_e + c0_plus_e*g_plus_e)

   ! Left boundary (anode)
   summ = 0.d0
   do j=1,nxl
       summ = summ + b_e(nx,j)*nen(j)
   enddo
   nen(nx) = (1.d0/e_e)*summ + (1.d0/e_e)*(cN_minus_e*g_minus_e + cN_plus_e*g_plus_e)

   !-----------------------------------
   ! ENERGY DENSITY AT BOUNDARIES
   !-----------------------------------
   ! Right boundary (cathode)
   summ = 0.d0
   do j=1,nxl
       summ = summ + b_W(0,j)*Wen(j)
   enddo
   Wen(0) = (1.d0/e_W)*summ + (1.d0/e_W)*(c0_minus_W*g_minus_W + c0_plus_W*g_plus_W)

   ! Left boundary (anode)
   summ = 0.d0
   do j=1,nxl
       summ = summ + b_W(nx,j)*Wen(j)
   enddo
   Wen(nx) = (1.d0/e_W)*summ + (1.d0/e_W)*(cN_minus_W*g_minus_W + cN_plus_W*g_plus_W)

   !-----------------------------------
   ! ELECTRIC POTENTIAL AT BOUNDARIES
   !-----------------------------------
   ! Right boundary (cathode)
   summ = 0.d0
   do j=1,nxl
       summ = summ + b_P(0,j)*Pn(j)
   enddo
   Pn(0) = (1.d0/e_P)*summ + (1.d0/e_P)*(c0_minus_P*g_minus_P + c0_plus_P*g_plus_P)

   ! Left boundary (anode)
   summ = 0.d0
   do j=1,nxl
       summ = summ + b_P(nx,j)*Pn(j)
   enddo
   Pn(nx) = (1.d0/e_P)*summ + (1.d0/e_P)*(cN_minus_P*g_minus_P + cN_plus_P*g_plus_P)

!====================================================================================
! FINAL DATA OUTPUT - SPATIAL PROFILES
!====================================================================================
   ! Write final spatial profiles for post-processing and analysis

   !-----------------------------------
   ! IONIZATION RATE PROFILE
   !-----------------------------------
   open(10,file='READ_REAL_TIME_Da_ne.dat')
   do i=0,nx
       write(10,3) r(i),R_i(i),H*R_i(i)
       ! Columns: radius, ionization rate, energy loss rate
   enddo
   close(10)

   !-----------------------------------
   ! ALL FIELD PROFILES
   !-----------------------------------
   open(10,file='READ_REAL_TIME_ni_ne_P_We_Te_E.dat')
   do i=0,nx
       write(10,7) r(i),nin(i),nen(i),Pn(i),Wen(i),Ten(i),En(i)
       ! Columns: radius, ion density, electron density, potential, energy density, temperature, E-field
   enddo
   close(10)

   !-----------------------------------
   ! FLUX PROFILES
   !-----------------------------------
   open(10,file='READ_REAL_TIME_qe_Je_Ji.dat')
   do i=0,nx
       write(10,4) r(i),qen(i),Jen(i),Jin(i)
       ! Columns: radius, energy flux, electron flux, ion flux
   enddo
   close(10)

   !-----------------------------------
   ! DRIFT AND DIFFUSION COMPONENTS
   !-----------------------------------
   open(10,file='READ_REAL_TIME_Je_Diff_Ji_Diff_Je_Drift_Ji_Drift.dat')
   do i=0,nx
       write(10,5) r(i),Je_Diff(i),Ji_Diff(i),Je_Drift(i),Ji_Drift(i)
       ! Columns: radius, electron diffusion, ion diffusion, electron drift, ion drift
   enddo
   close(10)

   !-----------------------------------
   ! SOURCE TERMS (INTERNAL POINTS ONLY)
   !-----------------------------------
   open(10,file='READ_REAL_TIME_fi_fe_fw_fP.dat')
   do i=1,nxl
       write(10,5) r(i),fi(i),fe(i),fw(i),fP(i)
       ! Columns: radius, ion source, electron source, energy source, charge source
   enddo
   close(10)

   !-----------------------------------
   ! ADVECTIVE COEFFICIENTS
   !-----------------------------------
   open(10,file='READ_REAL_TIME_a_i_a_e_a_w.dat')
   do i=1,nxl
       write(10,4) r(i),a_i(i),a_e(i),a_W(i)
       ! Columns: radius, ion advection, electron advection, energy advection
   enddo
   close(10)

   !-----------------------------------
   ! DIFFUSIVE COEFFICIENTS
   !-----------------------------------
   open(10,file='READ_REAL_TIME_v_i_v_e_v_w.dat')
   do i=1,nxl
       write(10,4) r(i),v_i(i),v_e(i),v_W(i)
       ! Columns: radius, ion diffusion, electron diffusion, energy diffusion
   enddo
   close(10)

!====================================================================================
! GPU MEMORY DEALLOCATION
!====================================================================================
   ! Free all GPU memory allocated for CUDA operations

   ! Pivot arrays for LU decomposition
   deallocate(devIpiv_i_d, devIpiv_e_d, devIpiv_w_d, devIpiv_P_d)

   ! System matrices on GPU
   deallocate(A_matrix_e_d)
   deallocate(A_matrix_i_d)
   deallocate(A_matrix_w_d)
   deallocate(A_matrix_P_d)

   ! Time history matrices on GPU
   deallocate(Bne_d,Bne_1_d,Bne_2_d,Bne_3_d)      ! Electron B matrices
   deallocate(Bni_d,Bni_1_d,Bni_2_d,Bni_3_d)      ! Ion B matrices
   deallocate(BnW_d,BnW_1_d,BnW_2_d,BnW_3_d)      ! Energy B matrices

   deallocate(Cne_d,Cne_1_d,Cne_2_d,Cne_3_d)      ! Electron C matrices
   deallocate(Cni_d,Cni_1_d,Cni_2_d,Cni_3_d)      ! Ion C matrices
   deallocate(CnW_d,CnW_1_d,CnW_2_d,CnW_3_d)      ! Energy C matrices

   ! Source term arrays on GPU
   deallocate(hi_d,he_d,hW_d,hP_d)

!====================================================================================
END PROGRAM plasma_cyl
!====================================================================================
!====================================================================================

!====================================================================================
!                              AUXILIARY SUBROUTINES
!====================================================================================

!====================================================================================
! CHEBYSHEV FIRST DERIVATIVE MATRIX
!====================================================================================
Subroutine Md1(d1,x,n)
    implicit none

    ! Arguments
    integer, intent (inout) :: n                    ! Number of Chebyshev points - 1
    double precision, intent (inout):: &
        d1(0:n,0:n),       & ! First derivative matrix
        x(0:n)               ! Chebyshev-Gauss-Lobatto points

    ! Local variables
    integer i,j
    double precision, dimension(:), allocatable :: c

    allocate (c(0:n))

    !-----------------------------------
    ! Set boundary coefficients
    !-----------------------------------
    ! c(i) = 2 for boundary points, 1 for interior points
    do i=0,n
        if (i==0.or.i==n) then
            c(i)=2.d0
        else
            c(i)=1.d0
        endif
    enddo

    !-----------------------------------
    ! Off-diagonal elements
    !-----------------------------------
    ! Standard Chebyshev differentiation matrix formula
    do i=0,n
        do j=0,n
            if(i.ne.j) then
                d1(i,j)=(c(i)/c(j))*(((-1.d0)**(i+j))/(x(i)-x(j)))
            endif
        enddo
    enddo

    !-----------------------------------
    ! Diagonal elements (interior points)
    !-----------------------------------
    do i=1,n-1
        d1(i,i)=-x(i)/( 2.d0*(1.d0-x(i)**2) )
    enddo

    !-----------------------------------
    ! Diagonal elements (boundary points)
    !-----------------------------------
    d1(0,0)=(2.d0*(dfloat(n)**2)+1.d0)/6.d0      ! Left boundary
    d1(n,n)=-d1(0,0)                             ! Right boundary

End Subroutine

!====================================================================================
! CHEBYSHEV SECOND DERIVATIVE MATRIX
!====================================================================================
Subroutine Md2(d2,x,n)
    implicit none

    ! Arguments
    integer, intent (inout) :: n                    ! Number of Chebyshev points - 1
    double precision, intent (inout):: &
        d2(0:n,0:n),       & ! Second derivative matrix
        x(0:n)               ! Chebyshev-Gauss-Lobatto points

    ! Local variables
    integer i,j
    double precision, dimension(:), allocatable :: c

    allocate (c(0:n))

    !-----------------------------------
    ! Set boundary coefficients
    !-----------------------------------
    do i=0,n
        if (i==0.or.i==n) then
            c(i)=2.d0
        else
            c(i)=1.d0
        endif
    enddo

    !-----------------------------------
    ! Off-diagonal elements (interior rows)
    !-----------------------------------
    do i=1,n-1
        do j=0,n
            if(i.ne.j) then
                d2(i,j)=((-1.d0)**(i+j)/c(j))*(x(i)**2+x(i)*x(j)-2.d0) &
                        /( (1.d0-x(i)**2)*((x(i)-x(j))**2))
            endif
        enddo
    enddo

    !-----------------------------------
    ! Diagonal elements (interior points)
    !-----------------------------------
    do i=1,n-1
        d2(i,i)=-((dfloat(n)**2-1.d0)*(1.d0-x(i)**2)+3.d0) &
                /(3.d0*((1.d0-x(i)**2)**2))
    enddo

    !-----------------------------------
    ! Boundary rows
    !-----------------------------------
    ! First row (i=0)
    do j=1,n
        d2(0,j)=(2.d0*((-1.d0)**j)/(3.d0*c(j)))*((2.d0*(dfloat(n)**2)+1.d0) &
                *(1.d0-x(j))-6.d0)/((1.d0-x(j))**2)
    enddo

    ! Last row (i=n)
    do j=0,n-1
        d2(N,j)=(2.d0*((-1.d0)**(j+n))/(3.d0*c(j))) &
                *((2.d0*(dfloat(n)**2)+1.d0)*(1.d0+x(j))-6.d0)/((1.d0+x(j))**2)
    enddo

    !-----------------------------------
    ! Corner elements
    !-----------------------------------
    d2(0,0)=(dfloat(n)**4-1.d0)/15.d0      ! Top-left corner
    d2(n,n)=d2(0,0)                         ! Bottom-right corner

End Subroutine

!====================================================================================
! GAUSSIAN WHITE NOISE GENERATOR (BOX-MULLER POLAR METHOD)
!====================================================================================
subroutine G_NOISE(nx,s1,s2)
    implicit none

    ! Arguments
    integer, intent (inout) :: nx           ! Number of grid points - 1
    double precision, intent (inout):: &
        s1(0:nx),         & ! First noise array
        s2(0:nx)            ! Second noise array

    ! Local variables
    integer :: i
    real :: w,x1,x2
    real :: mean,var

    ! Set noise statistics
    mean=0.0        ! Mean value
    var=1.0         ! Variance

    !-----------------------------------
    ! Generate noise at each grid point
    !-----------------------------------
    DO i=0,nx
        ! Box-Muller polar method
        DO
            ! Generate two uniform random numbers in [0,1]
            CALL RANDOM_NUMBER(x1)
            CALL RANDOM_NUMBER(x2)

            ! Transform to [-1,1]
            x1=2.0*x1-1.0
            x2=2.0*x2-1.0

            ! Calculate radius squared
            w=x1**2+x2**2

            ! Accept if inside unit circle
            if (w<1.0) exit
        END DO

        ! Box-Muller transformation
        w=sqrt(-2.0*log(w)/w)
        s1(i)=x1*w
        s2(i)=x2*w

        ! Apply mean and variance
        s1(i)=mean+sqrt(var)*s1(i)
        s2(i)=mean+sqrt(var)*s2(i)
    ENDDO

    return
end subroutine G_NOISE

!====================================================================================
! ALTERNATIVE WHITE NOISE GENERATOR
!====================================================================================
subroutine w_noise(nx,noise1,noise2)
    implicit none

    ! Arguments
    double precision, intent (inout):: noise1(0:nx),noise2(0:nx)
    integer, intent (inout) :: nx

    ! Local variables
    integer :: i
    real :: s,U1,U2,mean,variance

    ! Set noise statistics
    mean = 0
    variance = 1

    !-----------------------------------
    ! Generate noise using polar method
    !-----------------------------------
    DO i=0,nx
        DO
            call RANDOM_NUMBER(U1)  ! U1=[0,1]
            call RANDOM_NUMBER(U2)  ! U2=[0,1]
            U1 = 2*U1 - 1         ! U1=[-1,1]
            U2 = 2*U2 - 1         ! U2=[-1,1]
            s = U1*U1 + U2*U2
            if (s.LE.1) exit
        END DO

        ! Transform to Gaussian distribution
        noise1(i) = sqrt(-2*log(s)/s)*U1
        noise2(i) = sqrt(-2*log(s)/s)*U2
        noise1(i) = mean + sqrt(variance)*noise1(i)
        noise2(i) = mean + sqrt(variance)*noise2(i)
    ENDDO

    return
end subroutine w_noise

!====================================================================================
! ANIMATION DATA OUTPUT SUBROUTINE
!====================================================================================
subroutine ANIMATION(nit,nplot,nx,u0,v0,p0,E0,w0,x)
    implicit none

    ! Arguments
    integer, intent (inout) :: nit,nplot,nx
    double precision, intent (inout):: &
        u0(0:nx),         & ! Electron density
        v0(0:nx),         & ! Ion density
        p0(0:nx),         & ! Electric potential
        E0(0:nx),         & ! Energy density
        w0(0:nx),         & ! Temperature
        x(0:nx)             ! Spatial coordinate

    ! Local variables
    integer :: i,NCOUNT
    character :: EXT*4,NAMETEXT1*9,NCTEXT*5
    character :: OUTVEL(90000)*44
    character :: VELTEXT*35

    !-----------------------------------
    ! Set file paths and extensions
    !-----------------------------------
    EXT=".dat"
    VELTEXT="/home/leonardo/Desktop"   ! Output directory path

    !-----------------------------------
    ! Generate unique filename
    !-----------------------------------
    ! Add 10000 to avoid overwriting initial files
    NCOUNT=(nit/nplot)+10000
    WRITE(NCTEXT,'(I5)')NCOUNT
    NAMETEXT1=NCTEXT//EXT

    OUTVEL(NCOUNT)=VELTEXT//NAMETEXT1

    !-----------------------------------
    ! Write field data to file
    !-----------------------------------
    open(310,file=OUTVEL(NCOUNT))
    do i=0,nx
        write(310,5) x(i), u0(i), v0(i), p0(i), E0(i), w0(i)
        ! Columns: position, electron density, ion density, potential, energy, temperature
    enddo
    close(310)

    !-----------------------------------
    ! Format statement
    !-----------------------------------
5   format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)

    return
end subroutine ANIMATION

