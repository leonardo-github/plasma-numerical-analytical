!====================================================================================
!====================================================================================
!In this code it's solved a dimensionless warm plasma mathematical model

!----->  T ∂/∂t ne + (Pe ∇φ + ...) · ∇ne = ∇^2 ne + (...) + GAMA Pe (ni − ne) ne + Da neexp(-Ea/Te) (for electrons)

!-----> T β ∂/∂t ni − Pi ∇φ · ∇ni = ∇^2 ni + β Da ne exp(-Ea/Te) − GAMA Pi (ni − ne) ni  (for ions)


!----->   Energy...

!------------------->  ∇^2 φ = GAMA (ne − ni)                    (for poWential)

!====================================================================================
!====================================================================================
	program plasma_cyl
	use cusolverDn
	use cudafor
	implicit none
!====================================================================================

	!-----------------------------------
	integer :: t1,t2,ccr,istat,Lwork
	real(8), device, allocatable :: workspace_d(:)
	integer, device :: devInfo_d
	
	integer, device, dimension(:), allocatable :: devIpiv_i_d, devIpiv_e_d, devIpiv_w_d, devIpiv_P_d
	
	type(cusolverDnHandle) :: handle,handle2
	!-----------------------------------
	integer i,i0,i1,i2,i3,j,nx,nxl
	double precision :: k_delta,ppi,summ,S,R1,R2,CTF,Cr,r_1,r_2,x1,x2,x3,re1,re2,re3
	!Size for Gauss-Lobato base "x", source term "f", Source modified "h":
	double precision, dimension(:),allocatable :: &
               x,r,fe,fi,fP,fW,fTe,he,hi,hP,hW,hTe

	!Chebyshev Robin Condition Coefficients for e-lectrons,i-ons and P-oWential:
	double precision :: alfa_plus_e,alfa_minus_e,beta_plus_e,beta_minus_e,g_plus_e,g_minus_e
    double precision :: alfa_plus_i,alfa_minus_i,beta_plus_i,beta_minus_i,g_plus_i,g_minus_i
    double precision :: alfa_plus_P,alfa_minus_P,beta_plus_P,beta_minus_P,g_plus_P,g_minus_P
    double precision :: alfa_plus_W,alfa_minus_W,beta_plus_W,beta_minus_W,g_plus_W,g_minus_W
    double precision :: alfa_plus_T,alfa_minus_T,beta_plus_T,beta_minus_T,g_plus_T,g_minus_T

	double precision :: c0_plus_e,c0_minus_e,cN_plus_e,cN_minus_e,e_e
    double precision :: c0_plus_i,c0_minus_i,cN_plus_i,cN_minus_i,e_i
    double precision :: c0_plus_P,c0_minus_P,cN_plus_P,cN_minus_P,e_P
    double precision :: c0_plus_W,c0_minus_W,cN_plus_W,cN_minus_W,e_W
    double precision :: c0_plus_T,c0_minus_T,cN_plus_T,cN_minus_T,e_T

	!Chebyshev Dx, Dxx:
    double precision, dimension(:,:),allocatable :: d1x,d2x
	double precision, dimension(:,:),allocatable :: b_e,b_i,b_P,b_W,b_T
	double precision, dimension(:,:),allocatable :: DX_e,DXX_e,Dr_e,Drr_e
	double precision, dimension(:,:),allocatable :: DX_i,DXX_i,Dr_i,Drr_i
    double precision, dimension(:,:),allocatable :: DX_P,DXX_P,Dr_P,Drr_P
    double precision, dimension(:,:),allocatable :: DX_W,DXX_W,Dr_W,Drr_W
    double precision, dimension(:,:),allocatable :: DX_T,DXX_T,Dr_T,Drr_T

	!4th order (ADAMs BASHFORth/Backward differentiation)
	!Coefficients:
	double precision :: a0,a1,a2,a3,a4,b0,b1,b2,b3

	!Time Backward n^n+1, n^n, n^n-1, n^n-2, n^n-3 !electrons, ions and poWential
	!Matrixes:
	!poWential does not depends on time, ∇^2 φ(X) = GAMA [ ne(X,t) − ni(X,t) ]:
    double precision, dimension(:,:),allocatable :: A_matrix_e,A_matrix_inv_e
    double precision, dimension(:,:),allocatable :: A_matrix_i,A_matrix_inv_i
	double precision, dimension(:,:),allocatable :: A_matrix_P,A_matrix_inv_P
    double precision, dimension(:,:),allocatable :: A_matrix_W,A_matrix_inv_W
    double precision, dimension(:,:),allocatable :: A_matrix_T,A_matrix_inv_T

    double precision, dimension(:,:),allocatable :: Bne,Bne_1,Bne_2,Bne_3
    double precision, dimension(:,:),allocatable :: Bni,Bni_1,Bni_2,Bni_3
    double precision, dimension(:,:),allocatable :: BnW,BnW_1,BnW_2,BnW_3
    double precision, dimension(:,:),allocatable :: BnT,BnT_1,BnT_2,BnT_3

    double precision, dimension(:,:),allocatable :: Cne,Cne_1,Cne_2,Cne_3
    double precision, dimension(:,:),allocatable :: Cni,Cni_1,Cni_2,Cni_3
    double precision, dimension(:,:),allocatable :: CnW,CnW_1,CnW_2,CnW_3
    double precision, dimension(:,:),allocatable :: CnT,CnT_1,CnT_2,CnT_3

	!-------------------------------------------------------------------| GPU Arrays |
	
    double precision, device, dimension(:,:),allocatable :: A_matrix_e_d
    double precision, device, dimension(:,:),allocatable :: A_matrix_i_d
    double precision, device, dimension(:,:),allocatable :: A_matrix_W_d
    double precision, device, dimension(:,:),allocatable :: A_matrix_P_d
 
    double precision, device, dimension(:,:),allocatable :: Bne_d,Bne_1_d,Bne_2_d,Bne_3_d
    double precision, device, dimension(:,:),allocatable :: Bni_d,Bni_1_d,Bni_2_d,Bni_3_d
    double precision, device, dimension(:,:),allocatable :: BnW_d,BnW_1_d,BnW_2_d,BnW_3_d
  

    double precision, device, dimension(:,:),allocatable :: Cne_d,Cne_1_d,Cne_2_d,Cne_3_d
    double precision, device, dimension(:,:),allocatable :: Cni_d,Cni_1_d,Cni_2_d,Cni_3_d
    double precision, device, dimension(:,:),allocatable :: CnW_d,CnW_1_d,CnW_2_d,CnW_3_d

	double precision, device, dimension(:),allocatable :: hi_d,he_d,hW_d,hP_d
	
	!--------------------------------------------------------------------------------
	double precision, dimension(:),allocatable :: ne,nen,nen_1,nen_2,nen_3
	double precision, dimension(:),allocatable :: ni,nin,nin_1,nin_2,nin_3
	double precision, dimension(:),allocatable :: We,Wen,Wen_1,Wen_2,Wen_3
	double precision, dimension(:),allocatable :: Te,Ten,P,Pn,En,ME_e
	
    double precision, dimension(:),allocatable :: r_t,ni_t,ne_t,P_t,We_t,Te_t

    double precision, dimension(:),allocatable :: a_W,v_W
    double precision, dimension(:),allocatable :: a_e,v_e
    double precision, dimension(:),allocatable :: a_i,v_i
    double precision, dimension(:),allocatable :: Jen,Jin,qen,R_i
    double precision, dimension(:),allocatable :: Je_Diff,Ji_Diff,Je_Drift,Ji_Drift
    double precision, dimension(:),allocatable :: Dr_nen,Dr_nin,Dr_Wen,Dr_Ten,Dr_Pn
    double precision, dimension(:),allocatable :: Drr_nen,Drr_nin,Drr_Wen,Drr_Ten,Drr_Pn

!===============================================================================================================


! SUBROUTINE GAUSSIAN_NOISE

	double precision :: noise_factor
	double precision, dimension(:),allocatable :: &
               s1,s2,Dx_s1,Dx_s2

! SUBROUTINE ANIMATION

	integer :: nplot
	double precision :: deltatime

! TIME PROGRAM VARIABLES

	integer(8) :: nmax
	integer :: K,nit,nprint,kT
	double precision :: delta_e,delta_i,delta_P,delta_W,ht,time,t

!====================================================================================
    !DIMENSIONAL
    double precision :: &
    nu,tau,me,mi,de,Di,Ei,Te0,Tec,Ti0,Hei,ki0,N,n0,q0,J0,VRF,VDCRF,VDC,V0,L,R0,KB,KB1,e_0,perm_void,ME_0,Pg,Ng,Tg
    double precision :: &
    Pg_bar,Pg_Nmm2,kr,ks
    
    !DIMENSIONLESS
	double precision :: &
	Da,Pe,Pi,l3,B,GAMA,Y,H,xe,X0,D_o,Ea,D_r,K_s !where *Y* stands for lowercase gamma and *B* for lowercase beta

!====================================================================================
!          DimensionLESS parameters:
write(*,*)'Dimensional parameters'

write(*,*)' '
 r_1 = 0.5d0
 r_2 = 1.75d0
 L = r_2-r_1        ![cm]   interelectrode spacing
 R0= L!5.08d0       ![cm] Electrode radius

perm_void = 8.85d-14![c*V^-1*cm^-1
e_0= 1.6d-19        ![c]
KB = 8.621738d-5*e_0![eV/K]=[J/K]

tau= 1.d-9           ![s] dimensional characteristic time step
nu = 13.56d6        ![1/s]

Y = 0.046d0         ![ ] secondary electron emission coefficient

VDC= 200.d0!110.d0!77.4d0   ![V]        direct current voltage
V0 = 460.d0         ![V]        reference voltage

!TO calculate the the neutral particle density we must know 1 Torr = 1.3332236842 mbar = 1.3332E-4 N/mm^2 and P = N k T

    Pg_bar = 0.5d-3 !bar - Torr
   Pg_Nmm2 = 0.1d0*Pg_bar

 N = Pg_Nmm2/(273.d0*KB)!3.54E16!2.83E16![cm^-3] neutral species density, Which would imply the pressure of the camera  N = kT/P
 
 n0 = 4.d9!(perm_void*V0)/(e_0*L*L)  !4E9   !Reference particle density [cm^-3] 

Te0= 1.d0*e_0            ![J]        reference electron temperature but there is [1 K/8.621738E-5*e_0 eV]
Tec= 0.5d0*e_0      ![J]        electron temperature at cathode

Ei = 24.d0*e_0      ![J]        ionization rate activation energy
Hei= 15.578d0*e_0   ![J]        ionization enthalpy loss

ki0= 2.5d-7         ![cm^3/s]   ionization rate prefactorc
kr= 2.5d-7         ![cm^3/s]  recombination rate prefactorc
ks = 1.19d7

Di = 1.d2            ![cm^2/s]   ion diffusivity
de = 1.d6            ![cm^2/s]   electron reference diffusivity

me = 2.d5            ![cm^2/V s] electron mobility
mi = 2.d3            ![cm^2/V s] ion mobility


J0 = de*n0/L        ![] Characteristic current
q0 = J0*Hei    ![]Charactreristic Energy flux

ME_0= (3.d0/2.d0)*KB*Te0/KB


!          DimensionLESS parameters:
write(*,*)'Dimensionless parameters'

write(*,*)' '


Pe = me*V0/de;  Pi = mi*V0/Di;  B = de/Di;  l3 = 1.d0!(R0**2.d0)/(tau*de)!nu*(R0**2.d0)/de;

GAMA = e_0*n0*R0**2.d0/(perm_void*V0);  D_o = de*e_0/(me*KB*Te0/KB);

Ea = Ei/(KB*Te0/KB);  xe = e_0*V0/Hei;  H = Hei/((3.d0/2.d0)*KB*Te0/KB);  X0 = xe*H

Da = (ki0*N*R0**2.d0)/de;

D_r= (kr*n0*R0**2.d0)/de;

K_s = ks*R0/de


write(*,*)'Dimensionless parameters'
write(*,*)' '


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
!====================================================================================

! Chebyshev n-grid points:
! Chebyshev0n-grid points:
	nx = 169                !Number of Gauss-Lobatto points
	nxl = nx - 1
! Geometric parameter at the center of the domain:
	i0 = nint(nx/2.d0)      !Geometric center value for i

! Time marching:
	time = 200.d0 !Dimensionless Time of the simulation 10 mc s/1 mc s
 	  ht = 1.d-7   !Time step for stable calculation
	nmax =(time/ht)   !Total number of time steps!l3

	write(*,*)'Dimensionless time t/t_0=',time
    write(*,*)'Dimensionless step time dt=',ht
    write(*,*)'Total number of time steps nmax=',nmax
    write(*,*)'Total number of nodes nx=',nx
	write(*,*)'Dimensionless Potential at cathode -V_DC/V_0=',-VDC/V0
    write(*,*)'Dimensional Potential at cathode -V_DC=',-VDC
    write(*,*)'Dimensionless Energy at cathode Te_c/Te_0=',Te0/Te0
	write(*,*)' '

!write(*,*)'PRESS "ANY NUMBER" IF THE PARAMETERS ARE THE RIGTH ONES'
 !   read(*,*) S

	nprint=1         !Iteration number for printing in the screen

! Subroutine animation
      deltatime = 1.d0								!Time step for animation subroutine
      nplot = nint(deltatime/ht)					!Division of iteration for animation subroutine

! White noise:
   noise_factor = 0.d0
!====================================================================================

!High order TIME discretization COEFFICIENTS (ADAMs BASHFORth/Backward differentiation)

  !4th Order coefficients:

	a0 = 25.d0/12.d0;   a1 = -4.0d0;   a2 = 3.0d0;   a3 = -4.0d0/3.d0;   a4 = 1.0d0/4.d0

	b0 = 4.0d0;    b1 = -6.0d0;   b2 = 4.0d0;     b3 = -1.0d0
!====================================================================================
! Allocation of arrays:


    ! From axial coordinate *x* and source terms *f* and general source terms *h*  :
    !        x,      r,      fe,     fi,     fP,     fw,     he,     hi,     hP,     hw
	allocate(x(0:nx),r(0:nx),fe(nxl),fi(nxl),fP(nxl),fW(nxl),fTe(nxl),he(nxl),hi(nxl),hP(nxl),hW(nxl),hTe(nxl))

	! Matrix of derivatives from Chevyshev method <<-----------------------------------
     !       b_e,            b_i,            b_P,            b_W
	allocate(b_e(0:nx,1:nxl),b_i(0:nx,1:nxl),b_P(0:nx,1:nxl),b_W(0:nx,1:nxl),b_T(0:nx,1:nxl)) ! for electrons, ions and poWential

    allocate(d1x(0:nx,0:nx),d2x(0:nx,0:nx)) !general matrix of Chevyshev polinomials

	allocate(DX_e(nxl,nxl),DXX_e(nxl,nxl)) ! d/dx and d²/dx² for electron equation
	allocate(Dr_e(nxl,nxl),Drr_e(nxl,nxl)) ! d/dr and (1/r)d/dr + d²/dx² for electron equation
	allocate(DX_i(nxl,nxl),DXX_i(nxl,nxl)) ! d/dx and d²/dx² for ion equation
	allocate(Dr_i(nxl,nxl),Drr_i(nxl,nxl)) ! d/dr and (1/r)d/dr + d²/dx² for ion equation
	allocate(DX_P(nxl,nxl),DXX_P(nxl,nxl)) ! d/dx and d²/dx² for poWential equation
	allocate(Dr_P(nxl,nxl),Drr_P(nxl,nxl)) ! d/dr and (1/r)d/dr + d²/dx² for poWential equation
	allocate(DX_W(nxl,nxl),DXX_W(nxl,nxl)) ! d/dx and d²/dx² for Temperature equation
	allocate(Dr_W(nxl,nxl),Drr_W(nxl,nxl)) ! d/dr and (1/r)d/dr + d²/dx² for Temperature equation
    allocate(DX_T(nxl,nxl),DXX_T(nxl,nxl)) ! d/dx and d²/dx² for Temperature equation
	allocate(Dr_T(nxl,nxl),Drr_T(nxl,nxl)) ! d/dr and (1/r)d/dr + d²/dx² for Temperature equation

    ! For the Poisson equation:

     ! Electron FLUX, Ion FLUX, Energy Flux, ELectric FIELD
    allocate(Jen(0:nx),Jin(0:nx),qen(0:nx))
    !       Drift and Diff fluxes
    allocate(Je_Diff(0:nx),Ji_Diff(0:nx),Je_Drift(0:nx),Ji_Drift(0:nx))
    !From Adam Bashforth matrixes:
    !        A_matrix_e,         A_matrix_inv_e
	allocate(A_matrix_e(nxl,nxl),A_matrix_inv_e(nxl,nxl)) !electrons
    allocate(A_matrix_i(nxl,nxl),A_matrix_inv_i(nxl,nxl)) !ions
    allocate(A_matrix_P(nxl,nxl),A_matrix_inv_P(nxl,nxl)) !poWential
    allocate(A_matrix_W(nxl,nxl),A_matrix_inv_W(nxl,nxl)) !Temperature
   !         Bne,         Bne_1,         Bne_2          Bne_3
    allocate(Bne(nxl,nxl),Bne_1(nxl,nxl),Bne_2(nxl,nxl),Bne_3(nxl,nxl)) !electrons
    allocate(Bni(nxl,nxl),Bni_1(nxl,nxl),Bni_2(nxl,nxl),Bni_3(nxl,nxl)) !ions
    allocate(BnW(nxl,nxl),BnW_1(nxl,nxl),BnW_2(nxl,nxl),BnW_3(nxl,nxl)) !Temperature
    !        Cne,         Cne_1,         Cne_2,         Cne_3
	allocate(Cne(nxl,nxl),Cne_1(nxl,nxl),Cne_2(nxl,nxl),Cne_3(nxl,nxl)) !electrons
    allocate(Cni(nxl,nxl),Cni_1(nxl,nxl),Cni_2(nxl,nxl),Cni_3(nxl,nxl)) !ions
    allocate(CnW(nxl,nxl),CnW_1(nxl,nxl),CnW_2(nxl,nxl),CnW_3(nxl,nxl)) !Temperature
    
    
    !-------------------------------------------------------------------| GPU Allocate arrays |
    allocate(devIpiv_i_d(nxl), devIpiv_e_d(nxl), devIpiv_w_d(nxl), devIpiv_P_d(nxl))
    allocate(A_matrix_e_d(nxl,nxl))
    allocate(A_matrix_i_d(nxl,nxl))
    allocate(A_matrix_w_d(nxl,nxl))
    allocate(A_matrix_P_d(nxl,nxl))
    
        allocate(Bne_d(nxl,nxl),Bne_1_d(nxl,nxl),Bne_2_d(nxl,nxl),Bne_3_d(nxl,nxl)) !electrons
    allocate(Bni_d(nxl,nxl),Bni_1_d(nxl,nxl),Bni_2_d(nxl,nxl),Bni_3_d(nxl,nxl)) !ions
    allocate(BnW_d(nxl,nxl),BnW_1_d(nxl,nxl),BnW_2_d(nxl,nxl),BnW_3_d(nxl,nxl)) !Temperature
    
    	allocate(Cne_d(nxl,nxl),Cne_1_d(nxl,nxl),Cne_2_d(nxl,nxl),Cne_3_d(nxl,nxl)) !electrons
    allocate(Cni_d(nxl,nxl),Cni_1_d(nxl,nxl),Cni_2_d(nxl,nxl),Cni_3_d(nxl,nxl)) !ions
    allocate(CnW_d(nxl,nxl),CnW_1_d(nxl,nxl),CnW_2_d(nxl,nxl),CnW_3_d(nxl,nxl)) !Temperature
    
    allocate(hi_d(nxl),he_d(nxl),hW_d(nxl),hP_d(nxl))
    !------------------------------------------------------------------------------------------
    
    !        ne,      nen,      nen_1,      nen_2,      nen_3
	allocate(ne(0:nx),nen(0:nx),nen_1(0:nx),nen_2(0:nx),nen_3(0:nx)) !electrons
    allocate(ni(0:nx),nin(0:nx),nin_1(0:nx),nin_2(0:nx),nin_3(0:nx)) !ions
    allocate(We(0:nx),Wen(0:nx),Wen_1(0:nx),Wen_2(0:nx),Wen_3(0:nx))!Temperature
     allocate(P(0:nx),Pn(0:nx),Te(0:nx),Ten(0:nx),En(0:nx),ME_e(0:nx))

    allocate(a_W(1:nxl),v_W(1:nxl),a_e(1:nxl),v_e(1:nxl),a_i(1:nxl),v_i(1:nxl))

    allocate(Dr_nen(0:nx),Dr_nin(0:nx),Dr_Wen(0:nx),Dr_Ten(0:nx),Dr_Pn(0:nx))
    allocate(Drr_nen(0:nx),Drr_nin(0:nx),Drr_Wen(0:nx),Drr_Ten(0:nx),Drr_Pn(0:nx))
    allocate(R_i(0:nx))

! Withe noise:
	allocate(s1(0:nx),s2(0:nx),Dx_s1(0:nx),Dx_s2(0:nx))
    allocate(r_t(0:nx),ni_t(0:nx),ne_t(0:nx),P_t(0:nx),We_t(0:nx),Te_t(0:nx))

! Chebyshev base grid:

	ppi=dacos(-1.d0)  !!!!!!! in our case *pi* is 3.1416... !!!!!!!

	R1 = r_1/r_2              !The left electrode
                ! R1 < R2
	R2 = 1.d0!R0/L!1.d0              !The right

	CTF = 2.d0/(R2-R1)

	Cr = 1.d0             ! In Cr = 0 we are in flat paralell shaped electrodes

	do i = 0,nx
                x(i) = dcos(dfloat(i)*ppi/dfloat(nx))
                r(i) = ((R2 - R1)*x(i) + (R2 + R1))/2.d0
	end do

	
       re1 = 0.34d0;  re2 = 0.57d0;  re3 = 0.8d0

        x1 = (2.d0*re1 - R2 - R1)/(R2 - R1)
        x2 = (2.d0*re2 - R2 - R1)/(R2 - R1)
        x3 = (2.d0*re3 - R2 - R1)/(R2 - R1)

        i1 = nint(nx*dacos(x1)/ppi)
        i2 = nint(nx*dacos(x2)/ppi)
        i3 = nint(nx*dacos(x3)/ppi)

	
	
! Calling Chebyshev derivative matrix calculations:

	call Md1(d1x,x,nx)
	call Md2(d2x,x,nx)

 !====================================================================================
!                                   INITIAL CONDITION
 	open(2,file = 'ni_vs_t.dat')!ions at fixed point
    open(3,file = 'ne_vs_t.dat')!electrons at fixed point
    open(11,file = 'We_vs_t.dat')!Energy at fixed point
    open(17,file = 'Te_vs_t.dat')!Temperature at fixed point
    open(12,file = 'P_vs_t.dat') !potential at fixed point
    open(16,file = 'E_vs_t.dat') !Electric field at fixed point
    open(13,file = 'Je_vs_t.dat')!Je at fixed point
    open(14,file = 'Ji_vs_t.dat')!Ji at fixed point
    open(15,file = 'qe_vs_t.dat')!qe at fixed point
	!open(12,file='K_S1_Dx_s1.dat')	!Random numbers

      nit = 0!40000000
      t = 0.d0
      
      write(*,*)'nit=',nit
    write(*,*)'Press any number if the parameters are right'
    read(*,*)S
      
    !Initial condition truncated to the first run in N = 100,000,000 en carpeta Lin
      
    !  open(18,file = 'Desktop_10004.dat',status='old')
    !  do i=0,nx
    !  read(18,6) r_t(i),ne_t(i),ni_t(i),P_t(i),We_t(i),Te_t(i)
    !  enddo
    !  close(18)
           


   !     x(0) = - 1   WICH IS RIGTH IN SPATIAL coordinate
   !     x(nx) = + 1  WICH IS  LEFT IN SPATIAL coordinate
   
   
                      ME_e = 1.d0
                       Ten = ME_e
                        Pn = 0.d0
        DO i = 0,nx
        
                    nin(i) = (3d10/n0)*((1.d0 - r(i))**2.d0)*r(i)**2.d0 + (1d10/n0)
                    
                    nen(i) = (3d10/n0)*((1.d0 - r(i))**2.d0)*r(i)**2.d0 + (1d10/n0)
                    
                    Wen(i) = ME_e(i)*nen(i)
        ENDDO   
    

     nen_1 = nen  !ne^(n-1)
     nen_2 = nen  !ne^(n-2)
     nen_3 = nen  !ne^(n-3)

     nin_1 = nin  !ni^(n-1)
     nin_2 = nin  !ni^(n-2)
     nin_3 = nin  !ni^(n-3)

     Wen_1 = Wen  !Te^(n-1)
     Wen_2 = Wen  !Te^(n-2)
     Wen_3 = Wen  !Te^(n-3)
!=======================================================================================================================
                                CALL RANDOM_SEED()

				!-----------------------------
				CALL SYSTEM_CLOCK( count_rate = ccr  )
				
				call system_clock(t1)
				!-----------------------------
   ! DO K = 20000001,nmax ! STARTS THE LOOP TIME :
     DO K = 1,nmax ! STARTS THE LOOP TIME :
	
!                           VARIABLES FOR ERROR CALCULATION:
                                    nit = nit + 1
                                    t = nit*ht
	write(*,*) nit                 
!first derivative for general geometry (Coordinate Transformation Factor,CTF)

         En =-CTF*MATMUL(d1x,Pn)!  DxxPn = GAMA*(nen(i) - nin(i)); DxxPn = -DxEn

     Dr_nin = CTF*MATMUL(d1x,nin)
     Dr_nen = CTF*MATMUL(d1x,nen)
     Dr_Wen = CTF*MATMUL(d1x,Wen)
     Dr_Ten = CTF*MATMUL(d1x,Ten)

       qen = -(5.d0/(3.d0*H))*(Pe*En*Wen + (Ten/D_o)*Dr_Wen)

       Jen = -Pe*En*nen - (Ten/D_o)*Dr_nen

       Jin = (Pi*nin*En - Dr_nin)/B

!Here is calculated separately the Drift from the Diffusion contribution of the density current
   Je_Diff = -(Ten/D_o)*Dr_nen
   Ji_Diff = -Dr_nin/B

  Je_Drift = -Pe*En*nen
  Ji_Drift = Pi*nin*En/B
!=======================================================================================================
    !                               Withe noise generation
    call G_NOISE(nx,s1,s2)

	Dx_s1 = MATMUL(d1x,s1)
	Dx_s1 = noise_factor*Dx_s1
    !Dx_s2 = MATMUL(d1x,s2)
	!Dx_s2 = noise_factor*Dx_s2

	!write(12,*) K, s1(1), Dx_s1(1)
! Signal recording

	IF ((mod(nit,nprint).eq.0).or.(K.eq.1)) THEN
      write(2,4) t, nin(i1), nin(i2), nin(i3)
	  write(3,4) t, nen(i1), nen(i2), nen(i3)
     write(11,4) t, Wen(i1), Wen(i2), Wen(i3)
     write(17,4) t, Ten(i1), Ten(i2), Ten(i3)
     write(12,4) t, Pn(i1), Pn(i2), Pn(i3)
     write(16,4) t, En(i1), En(i2), En(i3)
     write(13,4) t, Jen(i1), Jen(i2), Jen(i3)
     write(14,4) t, Jin(i1), Jin(i2), Jin(i3)
     write(15,4) t, qen(i1), qen(i2), qen(i3)
	ENDIF
!=======================================================================================================




!                           BOUNDARY COEFFICIENTS:
!                                     right electrode  +1=1 U(0)=U(L)  : Right   CATHODE
!   the electron flux Je=-YJi is    Jen = -Pe*En*nen - (Dr_Ten/D_o)*nen - (Ten/D_o)*Dr_nen = - YJin
      alfa_plus_P = 1.d0;                        beta_plus_P = 0.d0;             g_plus_P = -VDC/V0
      alfa_plus_i = 0.d0;                        beta_plus_i = CTF ;             g_plus_i = 0.d0
      !alfa_plus_e =-Pe*En(0);                    beta_plus_e =0.d0;              g_plus_e = -Y*Ji_Drift(0)
      alfa_plus_e =-Pe*En(0);                    beta_plus_e =-CTF*Ten(0)/D_o;   g_plus_e = K_s*nen(0)-Y*Jin(0)
      alfa_plus_w = 1.d0;                        beta_plus_W = 0.d0;             g_plus_W = ME_e(0)*nen(0)
      alfa_plus_T = 1.d0;                        beta_plus_T = 0.d0;             g_plus_T = 1.d0

!Jen = -Pe*nen*En - (Wen/D_o)*Dxne
!                                     left electrode -1=0 U(nx)=U(0) :   Left   ANODE
      alfa_minus_P = 1.d0;        beta_minus_P = 0.d0;              g_minus_P = 0.d0
      alfa_minus_i = 0.d0;        beta_minus_i = CTF;               g_minus_i = 0.d0
      alfa_minus_e = 1.d0;        beta_minus_e = 0.d0;              g_minus_e = 0.d0
      alfa_minus_W = 1.d0;        beta_minus_W = 0.d0;              g_minus_W = 0.d0
    !(Pe*En*Wen + (Ten/D_o)*Dr_Wen)  
     !alfa_minus_W = -Pe*En(0);   beta_minus_W =-CTF*Ten(0)/D_o;    g_minus_W = 0.d0
      !alfa_minus_T = 1.d0;        beta_minus_T = 0.d0;              g_minus_T = 0.5d0
      alfa_minus_T = 0.d0;        beta_minus_T = (5.d0/3.d0)*CTF;   g_minus_T = -X0*En(nx)
!=====================================================================================
!                            Chevyshev COEFFICIENTS:

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

                    e_e = c0_plus_e*cN_minus_e - c0_minus_e*cN_plus_e
                    e_i = c0_plus_i*cN_minus_i - c0_minus_i*cN_plus_i
                    e_P = c0_plus_P*cN_minus_P - c0_minus_P*cN_plus_P
                    e_W = c0_plus_W*cN_minus_W - c0_minus_W*cN_plus_W
                    e_T = c0_plus_T*cN_minus_T - c0_minus_T*cN_plus_T

    do j=1,nxl
               b_e(0,j) = -c0_plus_e*beta_plus_e*d1x(0,j) - c0_minus_e*beta_minus_e*d1x(nx,j)!ne
               b_i(0,j) = -c0_plus_i*beta_plus_i*d1x(0,j) - c0_minus_i*beta_minus_i*d1x(nx,j)!ni
               b_P(0,j) = -c0_plus_P*beta_plus_P*d1x(0,j) - c0_minus_P*beta_minus_P*d1x(nx,j)!P
               b_W(0,j) = -c0_plus_W*beta_plus_W*d1x(0,j) - c0_minus_W*beta_minus_W*d1x(nx,j)!W
               b_T(0,j) = -c0_plus_T*beta_plus_T*d1x(0,j) - c0_minus_T*beta_minus_T*d1x(nx,j)!T
    enddo
    
    do j=1,nxl
              b_e(nx,j) = -cN_minus_e*beta_minus_e*d1x(nx,j) - cN_plus_e*beta_plus_e*d1x(0,j) !ne
              b_i(nx,j) = -cN_minus_i*beta_minus_i*d1x(nx,j) - cN_plus_i*beta_plus_i*d1x(0,j) !ni
              b_P(nx,j) = -cN_minus_P*beta_minus_P*d1x(nx,j) - cN_plus_P*beta_plus_P*d1x(0,j) !P
              b_W(nx,j) = -cN_minus_W*beta_minus_W*d1x(nx,j) - cN_plus_W*beta_plus_W*d1x(0,j) !W
              b_T(nx,j) = -cN_minus_T*beta_minus_T*d1x(nx,j) - cN_plus_T*beta_plus_T*d1x(0,j) !T
    enddo
!=========================================================================================

!                                      CHEVYSHEV DERIVATIVES MATRICXWA

!                                     dx = DX(i,j);  dxx = DXX(i,j)
DO i=1,nxl
     DO j=1,nxl
                DX_e(i,j) = d1x(i,j) + (1.d0/e_e)*(b_e(0,j)*d1x(i,0) + b_e(nx,j)*d1x(i,nx))
                DX_i(i,j) = d1x(i,j) + (1.d0/e_i)*(b_i(0,j)*d1x(i,0) + b_i(nx,j)*d1x(i,nx))
                DX_P(i,j) = d1x(i,j) + (1.d0/e_P)*(b_P(0,j)*d1x(i,0) + b_P(nx,j)*d1x(i,nx))
                DX_W(i,j) = d1x(i,j) + (1.d0/e_W)*(b_W(0,j)*d1x(i,0) + b_W(nx,j)*d1x(i,nx))
                DX_T(i,j) = d1x(i,j) + (1.d0/e_T)*(b_T(0,j)*d1x(i,0) + b_T(nx,j)*d1x(i,nx))

               DXX_e(i,j) = d2x(i,j) + (1.d0/e_e)*(b_e(0,j)*d2x(i,0) + b_e(nx,j)*d2x(i,nx))
               DXX_i(i,j) = d2x(i,j) + (1.d0/e_i)*(b_i(0,j)*d2x(i,0) + b_i(nx,j)*d2x(i,nx))
               DXX_P(i,j) = d2x(i,j) + (1.d0/e_P)*(b_P(0,j)*d2x(i,0) + b_P(nx,j)*d2x(i,nx))
               DXX_W(i,j) = d2x(i,j) + (1.d0/e_W)*(b_W(0,j)*d2x(i,0) + b_W(nx,j)*d2x(i,nx))
               DXX_T(i,j) = d2x(i,j) + (1.d0/e_T)*(b_T(0,j)*d2x(i,0) + b_T(nx,j)*d2x(i,nx))

!___________________    change of coordinates applied to derivatives    ___________________________

               Drr_e(i,j) = (CTF**2.d0)*DXX_e(i,j) + CTF*(Cr/r(i))*DX_e(i,j)
               Drr_i(i,j) = (CTF**2.d0)*DXX_i(i,j) + CTF*(Cr/r(i))*DX_i(i,j)
               Drr_P(i,j) = (CTF**2.d0)*DXX_P(i,j) + CTF*(Cr/r(i))*DX_P(i,j)
               Drr_W(i,j) = (CTF**2.d0)*DXX_W(i,j) + CTF*(Cr/r(i))*DX_W(i,j)
               Drr_T(i,j) = (CTF**2.d0)*DXX_T(i,j) + CTF*(Cr/r(i))*DX_T(i,j)

                Dr_e(i,j) = CTF*Dx_e(i,j)
                Dr_i(i,j) = CTF*Dx_i(i,j)
                Dr_P(i,j) = CTF*Dx_P(i,j)
                Dr_W(i,j) = CTF*DX_W(i,j)
                Dr_T(i,j) = CTF*DX_T(i,j)
          ENDDO
    ENDDO

!====================================================================================

!           ADAM BASHFORTH MATRIX CALCULATION FOR ELECRTRONS, IONS and POWenTIAL:

    do i=1,nxl
            do j=1,nxl
                        if (i.eq.j) then
                            k_delta=1.d0
                        else             !      K_delta = Kronecker delta = I
                            k_delta=0.d0
                        end if
                !ions
         v_i(i) = 1.d0
         a_i(i) = -Pi*En(i)

A_matrix_i(i,j) =B*l3*a0*k_delta - ht*v_i(i)*Drr_i(i,j)
       Bni(i,j) =B*l3*a1*k_delta - ht*b0*a_i(i)*Dr_i(i,j)
     Bni_1(i,j) =B*l3*a2*k_delta - ht*b1*a_i(i)*Dr_i(i,j)
     Bni_2(i,j) =B*l3*a3*k_delta - ht*b2*a_i(i)*Dr_i(i,j)
     Bni_3(i,j) =B*l3*a4*k_delta - ht*b3*a_i(i)*Dr_i(i,j)

!electrons
         v_e(i) = Ten(i)/D_o
         a_e(i) = Pe*En(i) + (1.d0/D_o)*Dr_Ten(i)

A_matrix_e(i,j) = l3*a0*k_delta - ht*v_e(i)*Drr_e(i,j)
       Bne(i,j) = l3*a1*k_delta - ht*b0*a_e(i)*Dr_e(i,j)
     Bne_1(i,j) = l3*a2*k_delta - ht*b1*a_e(i)*Dr_e(i,j)
     Bne_2(i,j) = l3*a3*k_delta - ht*b2*a_e(i)*Dr_e(i,j)
     Bne_3(i,j) = l3*a4*k_delta - ht*b3*a_e(i)*Dr_e(i,j)

!Temperature
         v_W(i) = (5.d0/3.d0)*(Ten(i)/D_o)
         a_W(i) = (5.d0/3.d0)*(Pe*En(i) + (1.d0/D_o)*Dr_Ten(i))

A_matrix_W(i,j) = l3*a0*k_delta - ht*v_W(i)*Drr_W(i,j)
       BnW(i,j) = l3*a1*k_delta - ht*b0*a_W(i)*Dr_W(i,j)
     BnW_1(i,j) = l3*a2*k_delta - ht*b1*a_W(i)*Dr_W(i,j)
     BnW_2(i,j) = l3*a3*k_delta - ht*b2*a_W(i)*Dr_W(i,j)
     BnW_3(i,j) = l3*a4*k_delta - ht*b3*a_W(i)*Dr_W(i,j)

!poWential
A_matrix_P(i,j) = Drr_P(i,j)

            enddo
    enddo


!----------------------------------------------------------------------------| CuSolver |

	A_matrix_e_d=A_matrix_e
	A_matrix_i_d=A_matrix_i
	A_matrix_w_d=A_matrix_w
	A_matrix_P_d=A_matrix_P
	
	istat=cusolverDnCreate(handle)
	

	istat=cusolverDnDgetrf_bufferSize(handle,nxl,nxl,A_matrix_e_d,nxl,Lwork)
	istat=cusolverDnDgetrf_bufferSize(handle,nxl,nxl,A_matrix_i_d,nxl,Lwork) 
	istat=cusolverDnDgetrf_bufferSize(handle,nxl,nxl,A_matrix_w_d,nxl,Lwork)  
	istat=cusolverDnDgetrf_bufferSize(handle,nxl,nxl,A_matrix_P_d,nxl,Lwork) 
	
	allocate(workspace_d(Lwork))
	
	istat=cusolverDnDgetrf(handle, nxl, nxl, A_matrix_e_d, nxl, workspace_d, devIpiv_e_d, devInfo_d)
	istat=cusolverDnDgetrf(handle, nxl, nxl, A_matrix_i_d, nxl, workspace_d, devIpiv_i_d, devInfo_d)
	istat=cusolverDnDgetrf(handle, nxl, nxl, A_matrix_w_d, nxl, workspace_d, devIpiv_w_d, devInfo_d)
	istat=cusolverDnDgetrf(handle, nxl, nxl, A_matrix_P_d, nxl, workspace_d, devIpiv_P_d, devInfo_d)
	
	! C^n = A^-1 x B^n:
	Bne_d=Bne
	istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_e_d, nxl, devIpiv_e_d, Bne_d, nxl, devInfo_d)
	Cne=Bne_d
	
	Bni_d=Bni
	istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_i_d, nxl, devIpiv_i_d, Bni_d, nxl, devInfo_d)
	Cni=Bni_d
	
	BnW_d=BnW
	istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, nxl, A_matrix_w_d, nxl, devIpiv_w_d, BnW_d, nxl, devInfo_d)
	CnW=BnW_d
	
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

! Matrix A inversion calculation  :

!call MATRIX_INVERSION(A_matrix_e,A_matrix_inv_e,nxl)
!call MATRIX_INVERSION(A_matrix_i,A_matrix_inv_i,nxl)
!call MATRIX_INVERSION(A_matrix_P,A_matrix_inv_P,nxl)
!call MATRIX_INVERSION(A_matrix_W,A_matrix_inv_W,nxl)

! C^n = A^-1 x B^n:
 !   Cne = MATMUL(A_matrix_inv_e,Bne)
!	Cni = MATMUL(A_matrix_inv_i,Bni)
!	CnW = MATMUL(A_matrix_inv_W,BnW)

!  C^(n-1) = A^-1 x B^(n-1):
!	Cne_1 = MATMUL(A_matrix_inv_e,Bne_1)
 !   Cni_1 = MATMUL(A_matrix_inv_i,Bni_1)
  !  CnW_1 = MATMUL(A_matrix_inv_W,BnW_1)

! C^(n-2) = A^-1 x B^(n-2):
!	Cne_2 = MATMUL(A_matrix_inv_e,Bne_2)
 !   Cni_2 = MATMUL(A_matrix_inv_i,Bni_2)
  !  CnW_2 = MATMUL(A_matrix_inv_W,BnW_2)

! E^(n-3) = A^-1 x B^(n-3):
!	Cne_3 = MATMUL(A_matrix_inv_e,Bne_3)
 !   Cni_3 = MATMUL(A_matrix_inv_i,Bni_3)
  !  CnW_3 = MATMUL(A_matrix_inv_W,BnW_3)

!=======================================================a================================
!                         SOURCE TERMS for  ION, ELECTRON, Temperature and POWenTIAL:

    DO i=1,nxl
!iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
      fi(i) = -GAMA*Pi*(nin(i) - nen(i))*nin(i) + B*Da*nen(i)*DEXP(-Ea/Ten(i)) + B*Dx_s1(i)- B*D_r*nen(i)*nin(i)
     v_i(i) = 1.d0
     a_i(i) = -Pi*En(i)

       hi(i) = ht*fi(i) &
             + v_i(i)*(ht/e_i)*(c0_minus_i*g_minus_i + c0_plus_i*g_plus_i)*(CTF**2.d0)*d2x(i,0) &
             + v_i(i)*(ht/e_i)*(cN_minus_i*g_minus_i + cN_plus_i*g_plus_i)*(CTF**2.d0)*d2x(i,nx) &
             + v_i(i)*(ht/e_i)*(c0_minus_i*g_minus_i + c0_plus_i*g_plus_i)*CTF*(Cr/r(i))*d1x(i,0) &
             + v_i(i)*(ht/e_i)*(cN_minus_i*g_minus_i + cN_plus_i*g_plus_i)*CTF*(Cr/r(i))*d1x(i,nx) &
+(b0+b1+b2+b3)*a_i(i)*(ht/e_i)*(c0_minus_i*g_minus_i + c0_plus_i*g_plus_i)*CTF*d1x(i,0) &
+(b0+b1+b2+b3)*a_i(i)*(ht/e_i)*(cN_minus_i*g_minus_i + cN_plus_i*g_plus_i)*CTF*d1x(i,nx)

!eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

     fe(i) = GAMA*Pe*(nin(i) - nen(i))*nen(i) + Da*nen(i)*DEXP(-Ea/Ten(i)) + Dx_s1(i)- D_r*nen(i)*nin(i)
    v_e(i) = Ten(i)/D_o
    a_e(i) = Pe*En(i) + (1.d0/D_o)*Dr_Ten(i)

         he(i) = ht*fe(i) &
               + v_e(i)*(ht/e_e)*(c0_minus_e*g_minus_e + c0_plus_e*g_plus_e)*(CTF**2.d0)*d2x(i,0) &
               + v_e(i)*(ht/e_e)*(cN_minus_e*g_minus_e + cN_plus_e*g_plus_e)*(CTF**2.d0)*d2x(i,nx) &
               + v_e(i)*(ht/e_e)*(c0_minus_e*g_minus_e + c0_plus_e*g_plus_e)*CTF*(Cr/r(i))*d1x(i,0) &
               + v_e(i)*(ht/e_e)*(cN_minus_e*g_minus_e + cN_plus_e*g_plus_e)*CTF*(Cr/r(i))*d1x(i,nx) &
 + (b0+b1+b2+b3)*a_e(i)*(ht/e_e)*(c0_minus_e*g_minus_e + c0_plus_e*g_plus_e)*CTF*d1x(i,0) &
 + (b0+b1+b2+b3)*a_e(i)*(ht/e_e)*(cN_minus_e*g_minus_e + cN_plus_e*g_plus_e)*CTF*d1x(i,nx)

!WWWWWWWWWWWWWWwwwWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

     fW(i) = (5.d0/3.d0)*GAMA*Pe*(nin(i) - nen(i))*Wen(i) - X0*Jen(i)*En(i)&
           - H*Da*nen(i)*DEXP(-Ea/Ten(i)) + H*D_r*nen(i)*nin(i)
    v_W(i) = (5.d0/3.d0)*(Ten(i)/D_o)
    a_W(i) = (5.d0/3.d0)*(Pe*En(i) + (1.d0/D_o)*Dr_Ten(i))

        hW(i) = ht*fW(i) &
              + v_W(i)*(ht/e_W)*(c0_minus_W*g_minus_W + c0_plus_W*g_plus_W)*(CTF**2.d0)*d2x(i,0) &
              + v_W(i)*(ht/e_W)*(cN_minus_W*g_minus_W + cN_plus_W*g_plus_W)*(CTF**2.d0)*d2x(i,nx) &
              + v_W(i)*(ht/e_W)*(c0_minus_W*g_minus_W + c0_plus_W*g_plus_W)*CTF*(Cr/r(i))*d1x(i,0) &
              + v_W(i)*(ht/e_W)*(cN_minus_W*g_minus_W + cN_plus_W*g_plus_W)*CTF*(Cr/r(i))*d1x(i,nx) &
+ (b0+b1+b2+b3)*a_W(i)*(ht/e_W)*(c0_minus_W*g_minus_W + c0_plus_W*g_plus_W)*CTF*d1x(i,0) &
+ (b0+b1+b2+b3)*a_W(i)*(ht/e_W)*(cN_minus_W*g_minus_W + cN_plus_W*g_plus_W)*CTF*d1x(i,nx)


!PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
      fP(i) = GAMA*(nen(i) - nin(i))

      hP(i) = fP(i) - (1.d0/e_P)*(c0_minus_P*g_minus_P + c0_plus_P*g_plus_P)*(CTF**2.d0)*d2x(i,0) &
                    - (1.d0/e_P)*(cN_minus_P*g_minus_P + cN_plus_P*g_plus_P)*(CTF**2.d0)*d2x(i,nx) &
                    - (1.d0/e_P)*(c0_minus_P*g_minus_P + c0_plus_P*g_plus_P)*CTF*(Cr/r(i))*d1x(i,0) &
                    - (1.d0/e_P)*(cN_minus_P*g_minus_P + cN_plus_P*g_plus_P)*CTF*(Cr/r(i))*d1x(i,nx)
    ENDDO

!========================================================================================

!                     UPTADE OF THE NEW VARIABLES  ne^(n+1), ni^(n+1), P^(n+1)
	
	istat=cusolverDnCreate(handle)
	
	
	
	hi_d=hi(1:nxl)
	istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, 1, A_matrix_i_d, nxl, devIpiv_i_d, hi_d, nxl, devInfo_d)
	ni(1:nxl)=hi_d
	
	he_d=he(1:nxl)
	istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, 1, A_matrix_e_d, nxl, devIpiv_i_d, he_d, nxl, devInfo_d)
	ne(1:nxl)=he_d

	hW_d=hW(1:nxl)
	istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, 1, A_matrix_W_d, nxl, devIpiv_i_d, hW_d, nxl, devInfo_d)
	We(1:nxl)=hW_d
	
	hP_d=hP(1:nxl)
	istat=cusolverDnDgetrs(handle,CUBLAS_OP_N, nxl, 1, A_matrix_P_d, nxl, devIpiv_P_d, hP_d, nxl, devInfo_d)
	P(1:nxl)=hP_d
	

	
	istat=cusolverDnDestroy(handle)
	
            !ni(1:nxl) = MATMUL(A_matrix_inv_i,hi(1:nxl))& !ion density at actual time
            
             ni(1:nxl)=ni(1:nxl)&
                      - MATMUL(Cni,nin(1:nxl))&
                      - MATMUL(Cni_1,nin_1(1:nxl))&
                      - MATMUL(Cni_2,nin_2(1:nxl))&
                      - MATMUL(Cni_3,nin_3(1:nxl))

	
	
            !ne(1:nxl) = MATMUL(A_matrix_inv_e,he(1:nxl))& !electron density at actual time
            
             ne(1:nxl)=ne(1:nxl)&
                      - MATMUL(Cne,nen(1:nxl))&
                      - MATMUL(Cne_1,nen_1(1:nxl))&
                      - MATMUL(Cne_2,nen_2(1:nxl))&
                      - MATMUL(Cne_3,nen_3(1:nxl))

	
	
           !We(1:nxl) = MATMUL(A_matrix_inv_W,hW(1:nxl))& !Electron Temperature at actual time
           
            
               We(1:nxl)=We(1:nxl)& 
                     
               		- MATMUL(CnW,Wen(1:nxl))&
                      - MATMUL(CnW_1,Wen_1(1:nxl))&
                      - MATMUL(CnW_2,Wen_2(1:nxl))&
                      - MATMUL(CnW_3,Wen_3(1:nxl))

          ME_e(1:nxl) = Wen(1:nxl)/nen(1:nxl)

            Te(1:nxl) = ME_e(1:nxl)


	
	
           ! P(1:nxl) = MATMUL(A_matrix_inv_P,hP(1:nxl))   !PoWential at actual time

	
!===================================================================================

!               Boundary values calculations ne, ni, P

	      summ = 0.d0
	do j=1,nxl
          summ = summ + b_i(0,j)*ni(j)
	enddo
	ni(0) = (1.d0/e_i)*summ + (1.d0/e_i)*(c0_minus_i*g_minus_i + c0_plus_i*g_plus_i)

	        summ=0.d0
	do j=1,nxl
          summ = summ + b_i(nx,j)*ni(j)
	enddo
    ni(nx) = (1.d0/e_i)*summ + (1.d0/e_i)*(cN_minus_i*g_minus_i + cN_plus_i*g_plus_i)

!-----------------------------------------------------------------------------------
        summ = 0.d0
	do j=1,nxl
          summ = summ + b_e(0,j)*ne(j)
	enddo
	ne(0) = (1.d0/e_e)*summ + (1.d0/e_e)*(c0_minus_e*g_minus_e + c0_plus_e*g_plus_e)

	      summ = 0.d0
	do j=1,nxl
          summ = summ + b_e(nx,j)*ne(j)
	enddo
	ne(nx)=(1.d0/e_e)*summ + (1.d0/e_e)*(cN_minus_e*g_minus_e + cN_plus_e*g_plus_e)

!-----------------------------------------------------------------------------------
        summ = 0.d0
	do j=1,nxl
          summ = summ + b_W(0,j)*We(j)
	enddo
	We(0) = (1.d0/e_W)*summ + (1.d0/e_W)*(c0_minus_W*g_minus_W + c0_plus_W*g_plus_W)

	      summ = 0.d0
	do j=1,nxl
          summ = summ + b_W(nx,j)*We(j)
	enddo
	We(nx)= (1.d0/e_W)*summ + (1.d0/e_W)*(cN_minus_W*g_minus_W + cN_plus_W*g_plus_W)

!-----------------------------------------------------------------------------------
        summ = 0.d0
	do j=1,nxl
          summ = summ + b_T(0,j)*Te(j)
	enddo
	Te(0) = (1.d0/e_T)*summ + (1.d0/e_T)*(c0_minus_T*g_minus_T + c0_plus_T*g_plus_T)

	      summ = 0.d0
	do j=1,nxl
          summ = summ + b_T(nx,j)*Te(j)
	enddo
	Te(nx)= (1.d0/e_T)*summ + (1.d0/e_T)*(cN_minus_T*g_minus_T + cN_plus_T*g_plus_T)
!-----------------------------------------------------------------------------------

        summ = 0.d0
	do j=1,nxl
          summ = summ + b_P(0,j)*P(j)
	enddo
    P(0) = (1.d0/e_P)*summ + (1.d0/e_P)*(c0_minus_P*g_minus_P + c0_plus_P*g_plus_P)

        summ = 0.d0
	do j=1,nxl
          summ = summ + b_P(nx,j)*P(j)
	enddo
    P(nx) = (1.d0/e_P)*summ + (1.d0/e_P)*(cN_minus_P*g_minus_P + cN_plus_P*g_plus_P)

!====================================================================================
! Error calculation ne :

	delta_e = MAXVAL(ABS(ne(1:nxl)-nen(1:nxl)))

	IF(mod(nit,nprint).eq.0) print 1,nit,delta_e,ne(i0),ni(i0),P(i0),We(i0)

! Subroutine animation for ne,ni,Te,P,E,Je,Ji:


!   Subroutine animation for Je_Diff,Je_Drift,Ji_Diff,Ji_Drift
!	IF(mod(nit,nplot).eq.0)THEN
!     call ANIMATION(nit,nplot,nx,Je_Diff,Je_Drift,Ji_Diff,Ji_Drift,x)
!	END IF

 ! variable re-assignation For Backward Adam Bashford Time discretization:

                        nen_3 = nen_2
                        nen_2 = nen_1
                        nen_1 = nen
                          nen = ne

                        nin_3 = nin_2
                        nin_2 = nin_1
                        nin_1 = nin
                          nin = ni

                        Wen_3 = Wen_2
                        Wen_2 = Wen_1
                        Wen_1 = Wen
                          Wen = We

                           Pn = P

                          Ten = Te

                         ME_e = Ten

                          R_i = Da*nen*DEXP(-Ea/Ten)
                          
          
       
	IF(mod(nit,nplot).eq.0)THEN
     call ANIMATION(nit,nplot,nx,nen,nin,Pn,Wen,Ten,r)
	END IF         
	
    !IF(mod(nit,nplot).eq.0)THEN
    ! call ANIMATION(nit,nplot,nx,Ten,Jen,Jin,qen,r)
	!END IF                   
                          

  ! FOR THE REAL TIME READING OF ne,ni,P,Te,E,Je,Ji :
    open(10,file='READ_REAL_TIME_Da_ne.dat')
	do i=0,nx
	  write(10,3) r(i),R_i(i),H*R_i(i)
	enddo
	close(10)

	!open(10,file='READ_REAL_TIME_Te_We.dat')
	!do i=0,nx
	!  write(10,3) r(i),Ten(i),Wen(i)
	!enddo
	!close(10)

  !    a)   ion and electron densities ( ni ne P E )
  !write(10,5) r(i),n0*nin(i),n0*nen(i),V0*Pn(i),Te0*Wen(i)
	open(10,file='READ_REAL_TIME_ni_ne_P_We_Te_E.dat')
	do i=0,nx
	  write(10,7) r(i),nin(i),nen(i),Pn(i),Wen(i),Ten(i),En(i)
	enddo
	close(10)

  !    b) E, qe, and Je, Ji
	!open(10,file='READ_REAL_TIME_qe_Je_Ji.dat')
	!do i=0,nx
	!  write(10,4) r(i),qen(i),Jen(i),Jin(i)
	!enddo
	!close(10)


! Format files

 1    format(1x,'N=',I8,'  Del=',E10.3,'  ne=',F12.4,'  ni=',F12.4,'  P=',F12.4,'  We=',F12.4)
 3    format(1x,E12.5,2x,E12.5,2x,E12.5)
 4    format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)
 5    format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)
 6    format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)
 7    format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)

!=================================================================================================================

	
	END DO!    --------------------->   END DO LOOP TIME
	!-----------------------------------
	CALL SYSTEM_CLOCK(t2)
	
	PRINT*, 'Time by loop (seconds):', (t2-t1)/float(ccr)
	!-----------------------------------
	
    close(2)!ions at fixed point
    close(3)!electrons at fixed point
    close(11)!energy at fixed point
    close(17)!Temperature at fixed point
    close(12) !poWential at fixed point
    close(16) !E at fixed point
    close(13) !Je at fixed point
    close(14) !Ji at fixed point
    close(15) !qe at fixed point
!========================================================================================================================

!                        FINAL Boundary values calculation

!iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
     summ=0.d0
	do j=1,nxl
          summ=summ+b_i(0,j)*nin(j)
	enddo
	nin(0)=(1.d0/e_i)*summ+(1.d0/e_i)*(c0_minus_i*g_minus_i+c0_plus_i*g_plus_i)

        summ=0.d0
	do j=1,nxl
          summ=summ+b_i(nx,j)*nin(j)
	enddo
	nin(nx)=(1.d0/e_i)*summ+(1.d0/e_i)*(cN_minus_i*g_minus_i+cN_plus_i*g_plus_i)
!eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
        summ=0.d0
	do j=1,nxl
          summ=summ+b_e(0,j)*nen(j)
	enddo
	nen(0)=(1.d0/e_e)*summ+(1.d0/e_e)*(c0_minus_e*g_minus_e+c0_plus_e*g_plus_e)

        summ=0.d0
	do j=1,nxl
          summ=summ+b_e(nx,j)*nen(j)
	enddo
	nen(nx)=(1.d0/e_e)*summ+(1.d0/e_e)*(cN_minus_e*g_minus_e+cN_plus_e*g_plus_e)
!TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
        summ = 0.d0
	do j=1,nxl
          summ = summ + b_W(0,j)*Wen(j)
	enddo
	Wen(0) = (1.d0/e_W)*summ + (1.d0/e_W)*(c0_minus_W*g_minus_W + c0_plus_W*g_plus_W)

	      summ = 0.d0
	do j=1,nxl
          summ = summ + b_W(nx,j)*Wen(j)
	enddo
	Wen(nx)= (1.d0/e_W)*summ + (1.d0/e_W)*(cN_minus_W*g_minus_W + cN_plus_W*g_plus_W)

!PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
      summ=0.d0
	do j=1,nxl
          summ=summ+b_P(0,j)*Pn(j)
	enddo
	Pn(0)=(1.d0/e_P)*summ+(1.d0/e_P)*(c0_minus_P*g_minus_P+c0_plus_P*g_plus_P)

        summ=0.d0
	do j=1,nxl
          summ=summ+b_P(nx,j)*Pn(j)
	enddo
	Pn(nx)=(1.d0/e_P)*summ+(1.d0/e_P)*(cN_minus_P*g_minus_P+cN_plus_P*g_plus_P)
                      

  ! FOR THE REAL TIME READING OF ne,ni,P,Te,E,Je,Ji :
    open(10,file='READ_REAL_TIME_Da_ne.dat')
	do i=0,nx
	  write(10,3) r(i),R_i(i),H*R_i(i)
	enddo
	close(10)

  !    a)   ion and electron densities ( ni ne P E )
  !write(10,5) r(i),n0*nin(i),n0*nen(i),V0*Pn(i),Te0*Wen(i)
	open(10,file='READ_REAL_TIME_ni_ne_P_We_Te_E.dat')
	do i=0,nx
	  write(10,7) r(i),nin(i),nen(i),Pn(i),Wen(i),Ten(i),En(i)
	enddo
	close(10)

  !    b) E, qe, and Je, Ji
	open(10,file='READ_REAL_TIME_qe_Je_Ji.dat')
	do i=0,nx
	  write(10,4) r(i),qen(i),Jen(i),Jin(i)
	enddo
	close(10)

  !    c)  Drift_Diff Je Ji
	open(10,file='READ_REAL_TIME_Je_Diff_Ji_Diff_Je_Drift_Ji_Drift.dat')
	do i=0,nx
	  write(10,5) r(i),Je_Diff(i),Ji_Diff(i),Je_Drift(i),Ji_Drift(i)
	enddo
	close(10)

  !    d) source  Terms
	open(10,file='READ_REAL_TIME_fi_fe_fw_fP.dat')
	do i=1,nxl
	  write(10,5) r(i),fi(i),fe(i),fw(i),fP(i)
	enddo
	close(10)

	  !    d) advective  Terms
	open(10,file='READ_REAL_TIME_a_i_a_e_a_w.dat')
	do i=1,nxl
	  write(10,4) r(i),a_i(i),a_e(i),a_W(i)
	enddo
	close(10)

		  !    d) diffusive  Terms
	open(10,file='READ_REAL_TIME_v_i_v_e_v_w.dat')
	do i=1,nxl
	  write(10,4) r(i),v_i(i),v_e(i),v_W(i)
	enddo
	close(10)


	!-------------------------------------------------------------------| GPU Allocate arrays |
	  deallocate(devIpiv_i_d(nxl), devIpiv_e_d(nxl), devIpiv_w_d(nxl), devIpiv_P_d(nxl))
   deallocate(A_matrix_e_d(nxl,nxl))
    deallocate(A_matrix_i_d(nxl,nxl))
    deallocate(A_matrix_w_d(nxl,nxl))
    deallocate(A_matrix_P_d(nxl,nxl))
    
        deallocate(Bne_d(nxl,nxl),Bne_1_d(nxl,nxl),Bne_2_d(nxl,nxl),Bne_3_d(nxl,nxl)) !electrons
    deallocate(Bni_d(nxl,nxl),Bni_1_d(nxl,nxl),Bni_2_d(nxl,nxl),Bni_3_d(nxl,nxl)) !ions
    deallocate(BnW_d(nxl,nxl),BnW_1_d(nxl,nxl),BnW_2_d(nxl,nxl),BnW_3_d(nxl,nxl)) !Temperature
    
    	deallocate(Cne_d(nxl,nxl),Cne_1_d(nxl,nxl),Cne_2_d(nxl,nxl),Cne_3_d(nxl,nxl)) !electrons
    deallocate(Cni_d(nxl,nxl),Cni_1_d(nxl,nxl),Cni_2_d(nxl,nxl),Cni_3_d(nxl,nxl)) !ions
    deallocate(CnW_d(nxl,nxl),CnW_1_d(nxl,nxl),CnW_2_d(nxl,nxl),CnW_3_d(nxl,nxl)) !Temperature
    
    deallocate(hi_d(nxl),he_d(nxl),hW_d(nxl),hP_d(nxl))
    !------------------------------------------------------------------------------------------
    
	END PROGRAM plasma_cyl

!====================================================================================
!====================================================================================

!	Subroutine MATRIX_INVERSION(alfa,alfa_inv,nl)
!	implicit none

!	integer, intent (inout) :: nl

!	double precision, intent (inout):: &
 !             alfa(nl,nl),alfa_inv(nl,nl)


!	integer :: M,LDA,LDVR,LWORK,INFO

!	double precision, dimension(:,:),allocatable :: &
 !              VR

!	double precision, dimension(:),allocatable :: &
 !              IPIV,WORK

!	LDA=nl; LDVR=nl; LWORK=nl*2; M=nl

!	allocate (IPIV(nl),VR(LDVR,nl),WORK(LWORK))

!	VR=alfa

!	call DGETRF(M,nl,VR,LDA,IPIV,INFO)
!	if (INFO.ne.0) then
!	write(*,*) "An error occurred on DGETRF"
!	endif

!	call DGETRI(nl,VR,LDA,IPIV,WORK,LWORK,INFO)
!	if (INFO.ne.0) then
!	write(*,*) "An error occurred on DGETRI"
!	endif

!	alfa_inv=VR

!	End Subroutine

!====================================================================================

	Subroutine Md1(d1,x,n)
	implicit none

	integer, intent (inout) :: n

	double precision, intent (inout):: &
              d1(0:n,0:n),x(0:n)

	integer i,j

	double precision, dimension(:), allocatable :: &
            c

	allocate (c(0:n))

	do i=0,n
	  if (i==0.or.i==n) then
	    c(i)=2.d0
	  else
	    c(i)=1.d0
	  endif
	enddo

	do i=0,n
	do j=0,n
	  if(i.ne.j) then
	    d1(i,j)=(c(i)/c(j))*(((-1.d0)**(i+j))/(x(i)-x(j)))
	  endif
	enddo
	enddo

	do i=1,n-1
	  d1(i,i)=-x(i)/( 2.d0*(1.d0-x(i)**2) )
	enddo

	d1(0,0)=(2.d0*(dfloat(n)**2)+1.d0)/6.d0
	d1(n,n)=-d1(0,0)

	End Subroutine

!====================================================================================

	Subroutine Md2(d2,x,n)
	implicit none

	integer, intent (inout) :: n

	double precision, intent (inout):: &
              d2(0:n,0:n),x(0:n)

	integer i,j

	double precision, dimension(:), allocatable :: &
            c

	allocate (c(0:n))

	do i=0,n
	  if (i==0.or.i==n) then
	    c(i)=2.d0
	  else
	    c(i)=1.d0
	  endif
	enddo

	do i=1,n-1
	do j=0,n
	  if(i.ne.j) then
	    d2(i,j)=((-1.d0)**(i+j)/c(j))*(x(i)**2+x(i)*x(j)-2.d0) &
                    /( (1.d0-x(i)**2)*((x(i)-x(j))**2))
	  endif
	enddo
	enddo

	do i=1,n-1
	  d2(i,i)=-((dfloat(n)**2-1.d0)*(1.d0-x(i)**2)+3.d0) &
                  /(3.d0*((1.d0-x(i)**2)**2))
	enddo

	do j=1,n
	  d2(0,j)=(2.d0*((-1.d0)**j)/(3.d0*c(j)))*((2.d0*(dfloat(n)**2)+1.d0) &
                  *(1.d0-x(j))-6.d0)/((1.d0-x(j))**2)
	enddo

	do j=0,n-1
	  d2(N,j)=(2.d0*((-1.d0)**(j+n))/(3.d0*c(j))) &
                  *((2.d0*(dfloat(n)**2)+1.d0)*(1.d0+x(j))-6.d0)/((1.d0+x(j))**2)
	enddo

	d2(0,0)=(dfloat(n)**4-1.d0)/15.d0
	d2(n,n)=d2(0,0)

	End Subroutine

!================================================================================

!==========================================
! GAUSSIAN NOISE (BOX-MULLER POLAR METHOD)
!==========================================
	subroutine G_NOISE(nx,s1,s2)
	implicit none

	integer, intent (inout) :: nx

	double precision, intent (inout):: &
	s1(0:nx),s2(0:nx)

	integer :: i

	real :: w,x1,x2
	real :: mean,var

	mean=0.0		!Mean
	var=1.0			!Variance

	DO i=0,nx
		DO
			CALL RANDOM_NUMBER(x1)
			CALL RANDOM_NUMBER(x2)
			x1=2.0*x1-1.0
			x2=2.0*x2-1.0
			w=x1**2+x2**2
			if (w<1.0) exit
		END DO

		w=sqrt(-2.0*log(w)/w)
		s1(i)=x1*w
		s2(i)=x2*w
		s1(i)=mean+sqrt(var)*s1(i)
		s2(i)=mean+sqrt(var)*s2(i)
	ENDDO

	return
	end subroutine G_NOISE
!===============================================================================================================

subroutine w_noise(nx,noise1,noise2)
	implicit none
double precision, intent (inout):: noise1(0:nx),noise2(0:nx)
integer, intent (inout) :: nx
integer :: i
real :: s,U1,U2,mean,variance

        mean = 0
    variance = 1
!----------------------------------------------------
DO i=0,nx
        DO
        call RANDOM_NUMBER(U1)!U1=[0,1]
        call RANDOM_NUMBER(U2)!U2=[0,1]
        U1 = 2*U1 - 1 !U1=[-1,1]
        U2 = 2*U2 - 1 !U2=[-1,1]
         s = U1*U1 + U2*U2
        if (s.LE.1) exit
        END DO

    noise1(i) = sqrt(-2*log(s)/s)*U1
    noise2(i) = sqrt(-2*log(s)/s)*U2
    noise1(i) = mean + sqrt(variance)*noise1(i)
    noise2(i) = mean + sqrt(variance)*noise2(i)
ENDDO

	return
	end subroutine w_noise

!====================================================================================
!====================================================================================
!====================================================================================
!=======================
!  ANIMATION DATA OUTPUT
!=======================
          !call ANIMATION(nit,nplot,nx,ne,ni,Te,We,P,r)
	subroutine ANIMATION(nit,nplot,nx,u0,v0,p0,E0,w0,x)
	implicit none

	integer, intent (inout) :: nit,nplot,nx

	double precision, intent (inout):: &
	u0(0:nx),v0(0:nx),p0(0:nx),E0(0:nx),w0(0:nx),x(0:nx)

	integer :: i,NCOUNT
	character :: EXT*4,NAMETEXT1*9,NCTEXT*5

!	character :: OUTVEL(90000)*34
!	character :: VELTEXT*25

	character :: OUTVEL(90000)*44
	character :: VELTEXT*35
!===============================================================================================================
! Address's data files
	EXT=".dat"
!	VELTEXT="/home/alfil/data/plasma/u"
            !/home/leonardo/Escritorio/Plasma_Numerical_analysis/Cul_Lin_2/fields/fields_1/
	VELTEXT="/home/leonardo/Desktop"
	!"/Users/alfil/Academic/data/plasma/u"
!===============================================================================================================
	NCOUNT=(nit/nplot)+10000
	WRITE(NCTEXT,'(I5)')NCOUNT
	NAMETEXT1=NCTEXT//EXT

	OUTVEL(NCOUNT)=VELTEXT//NAMETEXT1
!===============================================================================================================
! Fields

	open(310,file=OUTVEL(NCOUNT))
	do i=0,nx
	  write(310,5) x(i), u0(i), v0(i), p0(i), E0(i), w0(i)
	enddo
	close(310)
!===============================================================================================================
!===============================================================================================================
! Format files

 3    format(1x,E12.5,2x,E12.5,2x,E12.5)
 4    format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)
 5    format(1x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5,2x,E12.5)
!===============================================================================================================
	return
	end subroutine ANIMATION



