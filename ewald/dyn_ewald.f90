! --------------------------------------------------------------------------------------------------
! This is an auxiliary program that helps build the dynamical matrix for Ewald summation.
! It is written in Fortran and compiled so that speed is guaranteed.
! Hongze Xia, Sat Aug 23 18:19:39 2014
! f2py -c -m dyn_ewald dyn_ewald.pyf dyn_ewald.f90
! Wed 22 Apr 2015 15:21:57 AEST: subroutine dyn_abcm is added to this file
! --------------------------------------------------------------------------------------------------
    SUBROUTINE vffm(atoms,mass,charge,rmesh,kmesh,alpha,vol,qvec,dyn,N,nq,nr,nk)

    IMPLICIT NONE
    ! ============constants========= !
    real(8), parameter :: pi = 3.1415927
    complex(8), parameter :: j = (0,1)
    ! ============input============= !
    ! atoms.shape=(N,3) so are the meshes
    real(8), intent(in) :: atoms(N,3),rmesh(nr,3),kmesh(nk,3),qvec(nq,3)
    real(8), intent(in) :: mass(N),charge(N)
    real(8), intent(in) :: alpha,vol
    integer, intent(in) :: N,nr,nk,nq
    ! ============output============ !
    complex(8),dimension(nq,3*N,3*N),intent(out) :: dyn
    ! ============local============= !
    ! force constant tensor
    real(8), dimension(3,3) :: fc,fc1
    ! temporary variables
    real(8) :: r(3),r2,kr,k_q2,k2,k_q(3),k_qr,qr,q1,q2,expa2r2,masssqrt
    integer :: n0,n1,n2,a,b,nrr,nkk

    dyn = 0.0
    !    
    DO n0 = 1,nq
        !
        DO n1 = 1,N
            !
            q1 = charge(n1)
            DO n2 = 1,N
                !
                q2 = charge(n2)
                masssqrt = 1.0/SQRT(mass(n1)*mass(n2))
                ! real space
                DO nrr = 1,nr
                    !
                    r = atoms(n1,:)-atoms(n2,:)+rmesh(nrr,:)
                    qr = DOT_PRODUCT(r,qvec(n0,:))
                    r2 = DOT_PRODUCT(r,r)
                    IF (r2 .LT. 1.0D-4) CYCLE
                    ! Mind the Negative sign
                    expa2r2 = EXP(-r2*alpha*alpha)
                    fc = 0.0
                    DO a = 1,3
                        DO b = 1,3
                        !
                            IF (a .EQ. b) THEN
                                fc(a,a) = -2*(r(a)**2)*alpha*expa2r2/SQRT(pi)/r2 &
                                        *(2*(alpha**2) + 3/r2)
                                fc(a,a) = fc(a,a) + 2.0*alpha*expa2r2/SQRT(pi)/r2 &
                                        + ERFC(alpha*SQRT(r2))/r2**1.5*(1.0-3.0* &
                                        r(a)**2/r2) 
                            ELSE
                                fc(a,b) = 4.0*SQRT(pi)*(alpha**3)*(r2**1.5) + &
                                        6.0*SQRT(pi)*alpha*SQRT(r2) + 3.0*pi/ &
                                        expa2r2*ERFC(alpha*SQRT(r2))
                                fc(a,b) = -fc(a,b)*r(a)*r(b)*expa2r2/(pi*r2**2.5)
                            END IF
                        END DO
                    END DO
                    !
                    fc = fc*q1*q2/2.0
                    dyn(n0,3*n1-2:3*n1,3*n1-2:3*n1) = dyn(n0,3*n1-2:3*n1,3*n1-2:3*n1) - fc/mass(n1)
                    dyn(n0,3*n1-2:3*n1,3*n2-2:3*n2) = dyn(n0,3*n1-2:3*n1,3*n2-2:3*n2) + fc*EXP(-j*qr)*masssqrt
                END DO
                ! END real space
                !
                ! for reciprocal and dipole
                r = atoms(n1,:)-atoms(n2,:)
                !
    !             IF (n1 .EQ. n2) CYCLE
                ! reciprocal space
                DO nkk = 1,nk
                    !
                    kr = DOT_PRODUCT(kmesh(nkk,:),r)
                    k_q = kmesh(nkk,:)+qvec(n0,:)
                    ! examine the qvec so that it is not too small
                    IF (SQRT(DOT_PRODUCT(k_q,k_q)) .LT. 1.0D-9) THEN
                        IF (nq-n0 .gt. 0) THEN
                            k_q = k_q+1.0D-4*(qvec(n0+1,:)-k_q)
                        ELSE IF (n0 .eq. nq .and. nq .ne. 1) THEN
                            k_q = k_q+1.0D-4*(qvec(nq-1,:)-k_q)
                        ELSE IF (nq .eq. 1) THEN
                            k_q = k_q+1.0D-6
                        END IF
                    END IF
                    k2 = DOT_PRODUCT(kmesh(nkk,:),kmesh(nkk,:))
                    k_q2 = DOT_PRODUCT(k_q,k_q)
                    k_qr = DOT_PRODUCT(k_q,r)
                    DO a = 1,3
                        DO b = 1,3
                            fc(a,b) = k_q(a)*k_q(b)
                            fc1(a,b) = kmesh(nkk,a)*kmesh(nkk,b)
                        END DO
                    END DO
                    !
                    fc = fc*q1*q2*4.*pi/k_q2*EXP(-k_q2/4./alpha**2)/vol
                    dyn(n0,3*n1-2:3*n1,3*n2-2:3*n2) = dyn(n0,3*n1-2:3*n1,3*n2-2:3*n2) + fc*EXP(j*kr)*masssqrt
                    IF (k2 .GT. 1.0D-6) THEN
                        fc1 = fc1*q1*q2*4.*pi/k2*EXP(-k2/4./alpha**2)/vol
                        dyn(n0,3*n1-2:3*n1,3*n1-2:3*n1) = dyn(n0,3*n1-2:3*n1,3*n1-2:3*n1) - fc1/mass(n1)*EXP(j*kr)
                    END IF
                END DO
                ! END reciprocal space
                !
            END DO
        !
        END DO
        !
    END DO

    END SUBROUTINE vffm

    SUBROUTINE abcm(atoms,mass,charge,rmesh,kmesh,alpha,vol,qvec,dyn_abcm,N,NION,nq,nr,nk)

    IMPLICIT NONE
    ! ============constants========= !
    real(8), parameter :: pi = 3.1415927
    complex(8), parameter :: j = (0,1)
    ! ============input============= !
    ! atoms.shape=(N,3) so are the meshes
    real(8), intent(in) :: atoms(N,3),rmesh(nr,3),kmesh(nk,3),qvec(nq,3)
    ! mass.shape=(N,) so is charge, qvec.shape=(3,)
    real(8), intent(in) :: mass(NION),charge(N)
    real(8), intent(in) :: alpha,vol
    integer, intent(in) :: N,NION,nr,nk,nq
    ! ============output============ !
    complex(8),dimension(nq,3*NION,3*NION),intent(out) :: dyn_abcm
    ! ============local============= !
    ! force constant tensor
    real(8), dimension(3,3) :: fc,fc1
    ! temporary variables
    real(8) :: r(3),r2,kr,k_q2,k2,k_q(3),k_qr,qr,q1,q2,expa2r2
    integer :: n0,n1,n2,a,b,nrr,nkk
    ! mass matrix inverted
    real(8),dimension(3*NION,3*NION) :: mmat
    !temporary dyn
    complex(8),dimension(3*N,3*N) :: dyn
    !
    ! matrix inversion
    complex*16,allocatable,dimension(:,:)::S
    complex*16,allocatable,dimension(:)::WORK
    integer,allocatable,dimension(:)::IPIV
    integer info,error,M
    !
    mmat = 0.0
    DO n1 = 1,NION
        mmat(n1*3-2,n1*3-2) = 1./mass(n1)
        mmat(n1*3-1,n1*3-1) = 1./mass(n1)
        mmat(n1*3,n1*3) = 1./mass(n1)
    END DO
    !
    DO n0 = 1,nq
        !
        dyn = 0.0
        !
        DO n1 = 1,N
            !
            q1 = charge(n1)
            DO n2 = 1,N
                !
                q2 = charge(n2)
    !             masssqrt = 1.0/SQRT(mass(n1)*mass(n2))
                ! real space
                DO nrr = 1,nr
                    !
                    r = atoms(n1,:)-atoms(n2,:)+rmesh(nrr,:)
                    qr = DOT_PRODUCT(r,qvec(n0,:))
                    r2 = DOT_PRODUCT(r,r)
                    IF (r2 .LT. 1.0D-4) CYCLE
                    ! Mind the Negative sign
                    expa2r2 = EXP(-r2*alpha*alpha)
                    fc = 0.0
                    DO a = 1,3
                        DO b = 1,3
                        !
                            IF (a .EQ. b) THEN
                                fc(a,a) = -2*(r(a)**2)*alpha*expa2r2/SQRT(pi)/r2 &
                                        *(2*(alpha**2) + 3/r2)
                                fc(a,a) = fc(a,a) + 2.0*alpha*expa2r2/SQRT(pi)/r2 &
                                        + ERFC(alpha*SQRT(r2))/r2**1.5*(1.0-3.0* &
                                        r(a)**2/r2) 
                            ELSE
                                fc(a,b) = 4.0*SQRT(pi)*(alpha**3)*(r2**1.5) + &
                                        6.0*SQRT(pi)*alpha*SQRT(r2) + 3.0*pi/ &
                                        expa2r2*ERFC(alpha*SQRT(r2))
                                fc(a,b) = -fc(a,b)*r(a)*r(b)*expa2r2/(pi*r2**2.5)
                            END IF
                        END DO
                    END DO
                    !
                    fc = fc*q1*q2/2.0
                    dyn(3*n1-2:3*n1,3*n1-2:3*n1) = dyn(3*n1-2:3*n1,3*n1-2:3*n1) - fc
                    dyn(3*n1-2:3*n1,3*n2-2:3*n2) = dyn(3*n1-2:3*n1,3*n2-2:3*n2) + fc*EXP(-j*qr)
                END DO
                ! END real space
                !
                ! for reciprocal and dipole
                r = atoms(n1,:)-atoms(n2,:)
                !
    !             IF (n1 .EQ. n2) CYCLE
                ! reciprocal space
                DO nkk = 1,nk
                    !
                    kr = DOT_PRODUCT(kmesh(nkk,:),r)
                    k_q = kmesh(nkk,:)+qvec(n0,:)
                    ! examine the qvec so that it is not too small
                    IF (SQRT(DOT_PRODUCT(k_q,k_q)) .LT. 1.0D-9) THEN
                        IF (nq-n0 .gt. 0) THEN
                            k_q = k_q+1.0D-4*(qvec(n0+1,:)-k_q)
                        ELSE IF (n0 .eq. nq .and. nq .ne. 1) THEN
                            k_q = k_q+1.0D-4*(qvec(nq-1,:)-k_q)
                        ELSE IF (nq .eq. 1) THEN
                            k_q = k_q+1.0D-6
                        END IF
                    END IF
                    k2 = DOT_PRODUCT(kmesh(nkk,:),kmesh(nkk,:))
                    k_q2 = DOT_PRODUCT(k_q,k_q)
                    k_qr = DOT_PRODUCT(k_q,r)
                    DO a = 1,3
                        DO b = 1,3
                            fc(a,b) = k_q(a)*k_q(b)
                            fc1(a,b) = kmesh(nkk,a)*kmesh(nkk,b)
                        END DO
                    END DO
                    !
                    fc = fc*q1*q2*4.*pi/k_q2*EXP(-k_q2/4./alpha**2)/vol
                    dyn(3*n1-2:3*n1,3*n2-2:3*n2) = dyn(3*n1-2:3*n1,3*n2-2:3*n2) + fc*EXP(j*kr)
                    IF (k2 .GT. 1.0D-6) THEN
                        fc1 = fc1*q1*q2*4.*pi/k2*EXP(-k2/4./alpha**2)/vol
                        dyn(3*n1-2:3*n1,3*n1-2:3*n1) = dyn(3*n1-2:3*n1,3*n1-2:3*n1) - fc1*EXP(j*kr)
                    END IF
                END DO
                ! END reciprocal space
                !
            END DO
        !
        END DO
        ! matrix operations with LAPACK
        M = (N-NION)*3
        ALLOCATE(S(M,M),WORK(M),IPIV(M),stat=error)
        IF (error.ne.0)THEN
          PRINT *,"error:not enough memory"
          STOP
        END IF
        ! define S as the BC-BC matrix
        S(:,:) = dyn(3*NION+1:3*N,3*NION+1:3*N)
        ! LU factorisation
        CALL ZGETRF(M,M,S,M,IPIV,info)
        IF(info .ne. 0) THEN
         WRITE(*,*) "LU factorisation failed"
        END IF
        ! Invert it
        CALL ZGETRI(M,S,M,IPIV,WORK,M,info)
        ! ABCM operation
        dyn_abcm(n0,:,:) = dyn(1:3*NION,1:3*NION) - MATMUL( dyn(1:3*NION,3*NION+1:3*N),MATMUL( S,dyn(3*NION+1:3*N,1:3*NION) ) )
        !
        dyn_abcm(n0,:,:) = MATMUL(mmat,dyn_abcm(n0,:,:))
        !
        DEALLOCATE(S,IPIV,WORK,stat=error)
        IF (error.ne.0)THEN
          PRINT *,"error:fail to release memory"
          STOP
        END IF
    END DO
    !
    END SUBROUTINE abcm