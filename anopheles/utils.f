! Copyright (C) 2009  Anand Patil
! 
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.


! whereneg = []
! wherepos = []
! 
! for i in xrange(n):
!     whereneg.append(np.where(B[:,i]<0))
!     wherepos.append(np.where(B[:,i]>0))
! 
! for c in xrange(n_cycles):
!     for i in xrange(n):
!         y_ += B[:,i]*new_val[i]
!         
!         ub = np.min(y_[wherepos[i]]/B[wherepos[i],i]) if len(wherepos[i][0]) > 0 else 1.e6
!         lb = np.max(y_[whereneg[i]]/B[whereneg[i],i]) if len(whereneg[i][0]) > 0 else -1.e6
!         
!         if lb>ub:
!             raise np.linalg.LinAlgError, 'Something is fucked up.'
!         elif lb == ub:
!             new_val[i] = lb
!             continue
!         else:            
!             new_val[i] = pm.rtruncnorm(0,1,lb,ub)
!             if new_val[i]<lb or new_val[i]>ub:
!                 raise ValueError
!         y_ -= B[:,i]*new_val[i]

      SUBROUTINE lcm(B,y,nv,u,n,nc,ny,Bl,nneg,pf,nl,um,lop)
!
! lcm is for 'Linear constraint Metropolis'. Metropolis samples the elements of y, 
! in order, under the constraint that B*nv <= y. nc cycles are done. The 'likelihood'
! is evaluated based on the not-found observations as 
! -sum(nneg*log(logit^{-1}(Bl*nv))) -sum(nneg)*log(pf), and the Metropolis acceptance
! is chosen based on the uniform random variables um.
!
cf2py intent(hide) ny, n, c, nl
cf2py intent(inplace) nv
      DOUBLE PRECISION B(ny,n), u(n,nc), y(ny), nv(n), lop(nl)
      DOUBLE PRECISION Bl(nl,n), um(n,nc), nneg(nl), pf
      DOUBLE PRECISION lb, ub, lb_, ub_, na, nb, u_, lpf
      DOUBLE PRECISION sqrt2, thisb, lopp(nl), llr, nvp, dev
      INTEGER ny, n, nc, c, i, j,ifault
      
      sqrt2 = dsqrt(2.0D0)
      lpf = dlog(pf)
      
      do c=1,nc
          do i=1,n
!               Figure out upper and lower bounds
              ub = 1.0D6
              lb = -1.0D6
              do j=1,ny
                  thisb = B(j,i)
                  y(j)=y(j)+thisb*nv(i)
                  if (thisb.GT.0.0D0) then
                      ub_ = y(j)/thisb
                      if (ub_.LT.ub) then
                          ub = ub_
                      end if
                  else if (thisb.LT.0.0D0) then
                      lb_ = y(j)/thisb
                      if (lb_.GT.lb) then
                          lb = lb_
                      end if
                  end if
              end do
                            
              if (lb.EQ.ub) then
                  nv(i) = lb
                  continue
              end if
              
!               Draw truncated normal and store
              na = 0.5D0*(1.0D0+derf(lb/sqrt2))
              nb = 0.5D0*(1.0D0+derf(ub/sqrt2))
              u_ = u(i,c)
              u_ = na + (nb-na)*u_
              ifault=0
              if (u_.EQ.1.0D0) then
                  u_ = ub
              else if (u_.EQ.0.0D0) then
                  u_ = lb
              else
                  CALL ppnd16(u_,ifault)
              end if
              
!               Evaluate log-likelihood ratio and new logit-probability.
              nvp = u_
              dev = nvp - nv(i)
              llr = 0.0D0
              do j=1,nl
                  lopp(j)=lop(j)+Bl(j,i)*dev
                  if (lopp(j).GT.0.0D0) then
                      llr = llr + nneg(j) * lpf
                  end if
                  if (lop(j).GT.0.0D0) then
                      llr=llr - nneg(j) * lpf
                  end if 
!                   llr=llr+nneg(j)*(dlog(1.0D0+dexp(-lop(j)))
!      *              -dlog(1.0D0+dexp(-lopp(j))))
              end do
!               print *,i,llr,um(i,c)
              
!               M-H acceptance?
              if (dlog(um(i,c)).LE.llr) then
!                   print *,'accepted',i,nv(i),llr
                  nv(i)=nvp
                  do j=1,nl
                      lop(j)=lopp(j)
                  end do
!               else
!                   print *,'rejected',llr
              end if

!               print *,i,lb,ub,na,nb,nv(i),u_,ifault
!               Bookkeeping on linear constraints.              
              do j=1,ny
                  y(j)=y(j)-B(j,i)*nv(i)
              end do              
          end do
      end do
      
      RETURN
      END
      

      SUBROUTINE lcg(B, y, nv, u, n, nc, ny)
!
! lcg is for 'Linear constraint Gibbs'. Gibbs samples the elements of y, in order,
! under the constraint that B*nv <= y. nc cycles are done.
!
cf2py intent(hide) ny, n, c      
cf2py intent(inplace) nv
      DOUBLE PRECISION B(ny,n), u(n,nc), y(ny), nv(n)
      DOUBLE PRECISION lb, ub, lb_, ub_, na, nb, u_
      DOUBLE PRECISION sqrt2, thisb
      INTEGER ny, n, nc, c, i, j,ifault
      
      sqrt2 = dsqrt(2.0D0)
      
      do c=1,nc
          do i=1,n
!               Figure out upper and lower bounds
              ub = 1.0D6
              lb = -1.0D6
              do j=1,ny
                  thisb = B(j,i)
                  y(j)=y(j)+thisb*nv(i)
                  if (thisb.GT.0.0D0) then
                      ub_ = y(j)/thisb
                      if (ub_.LT.ub) then
                          ub = ub_
                      end if
                  else if (thisb.LT.0.0D0) then
                      lb_ = y(j)/thisb
                      if (lb_.GT.lb) then
                          lb = lb_
                      end if
                  end if
              end do
                            
              if (lb.EQ.ub) then
                  nv(i) = lb
                  continue
              end if
              
!               Draw truncated normal and store
              na = 0.5D0*(1.0D0+derf(lb/sqrt2))
              nb = 0.5D0*(1.0D0+derf(ub/sqrt2))
              u_ = u(i,c)
              u_ = na + (nb-na)*u_
              ifault=0
              if (u_.EQ.1.0D0) then
                  u_ = ub
              else if (u_.EQ.0.0D0) then
                  u_ = lb
              else
                  CALL ppnd16(u_,ifault)
              end if
              nv(i) = u_
!               print *,i,lb,ub,na,nb,nv(i),u_,ifault
              
              do j=1,ny
                  y(j)=y(j)-B(j,i)*nv(i)
              end do              
          end do
      end do
      
      RETURN
      END



      SUBROUTINE logsum(x, nx, s)
cf2py intent(hide) nx
cf2py intent(out) s
cf2py threadsafe
      IMPLICIT NONE
      DOUBLE PRECISION x(nx), s, diff, li
      INTEGER nx, i
      PARAMETER (li=709.78271289338397)
      
      s = x(1)
      
      do i=2,nx
!           If x(i) swamps the sum so far, ditch the sum so far.
          diff = x(i)-s
          if (diff.GE.li) then
              s = x(i)
          else
              s = s + dlog(1.0D0+dexp(x(i)-s))
          end if
      end do

      RETURN
      END


      SUBROUTINE logsum2(z,x,y)
      
! Computes z <- x + (-1)^s * y.

      IMPLICIT NONE
      DOUBLE PRECISION x,y,z,d,li,ni
      PARAMETER (li=709.78271289338397)
      PARAMETER (ni=-1.7976931348623157d308)      
      
!     If y swamps x, return y.
      d = y-x
      if ((x.LE.ni).AND.(y.LE.ni)) then
           z = ni
      else
          if (d.GE.li) then
              z = y
          else
             z = x + dlog(1.0D0+dexp(d))
          end if
      end if

      RETURN
      END


      SUBROUTINE mahal(c,x,y,symm,dd,a,l,s,nx,ny,nd,cmin,cmax)
cf2py intent(hide) nx,ny,nd
cf2py intent(inplace) c
cf2py threadsafe
      DOUBLE PRECISION x(nx,nd), y(ny,nd), s(nd,nd), l(nd)
      DOUBLE PRECISION c(nx,ny), dev(nd), this, a, tdev(nd)
      DOUBLE PRECISION dd, rem, GA, prefac, snu
      INTEGER i,j,k,m,nx,ny,nd,cmin,cmax
      LOGICAL symm
      
      dd = 2.0D0
      N = 1
      ! FIXME: Double-check this GA.
      GA = 1.0D0
            
      prefac = 0.5D0 ** (dd-1.0D0) / GA
 
      snu = DSQRT(dd) * 2.0D0
      fl = DINT(dd)
      rem = dd - fl
      
!       DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
      EXTERNAL DGEMV
      
      do j=cmin+1,cmax     
          if (symm) then

                c(j,j)=a*a

                do i=1,j-1
                    do k=1,nd
                        dev(k) = x(i,k) - y(j,k)
                    end do

!               DGEMV that is guaranteed to be single-threaded
                    do k=1,nd
                        tdev(k) = 0.0D0                  
                        do m=1,nd
                          tdev(k)=tdev(k)+s(m,k)*dev(m)
                        end do
                    end do

                    this = 0.0D0
                    do k=1,nd
                        this = this - tdev(k)*tdev(k)/l(k)
                    end do

                  this = dsqrt(this) * snu
                  CALL RKBESL(this,rem,fl+1,1,BK,N)
                  c(i,j) = prefac*(this**diff_degree)*BK(fl+1)

                end do              
                
          else
          
              do i=1,nx
                            
                  do k=1,nd
                      dev(k) = x(i,k) - y(j,k)
                  end do
              
!               DGEMV that is guaranteed to be single-threaded
                  do k=1,nd
                      tdev(k) = 0.0D0                  
                      do m=1,nd
                        tdev(k)=tdev(k)+s(m,k)*dev(m)
                      end do
                  end do
              
                  this = 0.0D0
                  do k=1,nd
                      this = this - tdev(k)*tdev(k)/l(k)
                  end do
              
                  this = dsqrt(this) * snu
                  CALL RKBESL(this,rem,fl+1,1,BK,N)
                  c(i,j) = prefac*(this**diff_degree)*BK(fl+1)

              end do
          end if
      end do
      
      RETURN 
      END



      SUBROUTINE mod_mahal(c,ds,x,y,symm,a,l,s,cf,nx,ny,nd,cmin,cmax
     1,BK, diff_degree, GA, N)
cf2py intent(hide) nx,ny,nd,BK
cf2py intent(inplace) c
cf2py intent(hide) diff_degree = 2.0
cf2py intent(hide) GA = 1.0
cf2py intent(hide) N = 2
cf2py threadsafe
      DOUBLE PRECISION x(nx,nd), y(ny,nd), s(nd+1,nd+1), l(nd+1)
      DOUBLE PRECISION c(nx,ny), dev(nd+1), this, a, tdev(nd+1)
      DOUBLE PRECISION ds(nx,ny)
      DOUBLE PRECISION cf, diff_degree, rem, GA, prefac, snu
      DOUBLE PRECISION BK(15)
      INTEGER i,j,k,m,nx,ny,nd,cmin,cmax,N,fl
      LOGICAL symm
      
!       DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
      EXTERNAL DGEMV
      
      diff_degree = 1.0D0
      N = 1
      GA = 1.0D0
            
      prefac = 0.5D0 ** (diff_degree-1.0D0) / GA
 
      snu = DSQRT(diff_degree) * 2.0D0
      fl = DINT(diff_degree)
      rem = diff_degree - fl
      
!       print *,snu,fl,rem,diff_degree,prefac,GA      
      
      
      do j=cmin+1,cmax     
          if (symm) then
 
                c(j,j)=a*a
 
                do i=1,j-1
                    dev(1) = ds(i,j)
                    do k=1,nd
                        dev(k+1) = x(i,k) - y(j,k)
                    end do
 
!               DGEMV that is guaranteed to be single-threaded
                    do k=1,nd+1
                        tdev(k) = 0.0D0                  
                        do m=1,nd+1
                          tdev(k)=tdev(k)+s(m,k)*dev(m)
                        end do
                    end do
 
                    this = 0.0D0
                    do k=1,nd+1
                        this = this + tdev(k)*tdev(k)/l(k)
                    end do
                    
!                     this = dsqrt(this) * snu
!                     CALL RKBESL(this,rem,fl+1,1,BK,N)
!                     this = prefac*(this**diff_degree)*BK(fl+1)
                    this = dexp(-dsqrt(this))
                    c(i,j) = (this*(1.0D0-cf)+cf)*a*a
 
 
                end do              
                
          else
          
              do i=1,nx
                        
                  dev(1) = ds(i,j)    
                  do k=1,nd
                      dev(k+1) = x(i,k) - y(j,k)
                  end do
              
!               DGEMV that is guaranteed to be single-threaded
                  do k=1,nd+1
                      tdev(k) = 0.0D0                  
                      do m=1,nd+1
                        tdev(k)=tdev(k)+s(m,k)*dev(m)
                      end do
                  end do
              
                  this = 0.0D0
                  do k=1,nd+1
                      this = this + tdev(k)*tdev(k)/l(k)
                  end do
              
!                   this = dsqrt(this) * snu
!                   CALL RKBESL(this,rem,fl+1,1,BK,N)
!                   this = prefac*(this**diff_degree)*BK(fl+1)
                  this = dexp(-dsqrt(this))
                  c(i,j) = (this*(1.0D0-cf)+cf)*a*a
 
 
              end do
          end if
      end do
      
      RETURN 
      END

!
      SUBROUTINE RKBESL(X,ALPHA,NB,IZE,BK,NCALC)
cf2py intent (out) bk
C-------------------------------------------------------------------
C
C  This FORTRAN 77 routine calculates modified Bessel functions
C  of the second kind, K SUB(N+ALPHA) (X), for non-negative
C  argument X, and non-negative order N+ALPHA, with or without
C  exponential scaling.
C
C  Explanation of variables in the calling sequence
C
C  Description of output values ..
C
C X     - Working precision non-negative real argument for which
C         K's or exponentially scaled K's (K*EXP(X))
C         are to be calculated.  If K's are to be calculated,
C         X must not be greater than XMAX (see below).
C ALPHA - Working precision fractional part of order for which 
C         K's or exponentially scaled K's (K*EXP(X)) are
C         to be calculated.  0 .LE. ALPHA .LT. 1.0.
C NB    - Integer number of functions to be calculated, NB .GT. 0.
C         The first function calculated is of order ALPHA, and the 
C         last is of order (NB - 1 + ALPHA).
C IZE   - Integer type.  IZE = 1 if unscaled K's are to be calculated,
C         and 2 if exponentially scaled K's are to be calculated.
C BK    - Working precision output vector of length NB.  If the
C         routine terminates normally (NCALC=NB), the vector BK
C         contains the functions K(ALPHA,X), ... , K(NB-1+ALPHA,X),
C         or the corresponding exponentially scaled functions.
C         If (0 .LT. NCALC .LT. NB), BK(I) contains correct function
C         values for I .LE. NCALC, and contains the ratios
C         K(ALPHA+I-1,X)/K(ALPHA+I-2,X) for the rest of the array.
C NCALC - Integer output variable indicating possible errors.
C         Before using the vector BK, the user should check that 
C         NCALC=NB, i.e., all orders have been calculated to
C         the desired accuracy.  See error returns below.
C
C
C*******************************************************************
C*******************************************************************
C
C Explanation of machine-dependent constants
C
C   beta   = Radix for the floating-point system
C   minexp = Smallest representable power of beta
C   maxexp = Smallest power of beta that overflows
C   EPS    = The smallest positive floating-point number such that 
C            1.0+EPS .GT. 1.0
C   XMAX   = Upper limit on the magnitude of X when IZE=1;  Solution 
C            to equation:
C               W(X) * (1-1/8X+9/128X**2) = beta**minexp
C            where  W(X) = EXP(-X)*SQRT(PI/2X)
C   SQXMIN = Square root of beta**minexp
C   XINF   = Largest positive machine number; approximately
C            beta**maxexp
C   XMIN   = Smallest positive machine number; approximately
C            beta**minexp
C
C
C     Approximate values for some important machines are:
C
C                          beta       minexp      maxexp      EPS
C
C  CRAY-1        (S.P.)      2        -8193        8191    7.11E-15
C  Cyber 180/185 
C    under NOS   (S.P.)      2         -975        1070    3.55E-15
C  IEEE (IBM/XT,
C    SUN, etc.)  (S.P.)      2         -126         128    1.19E-7
C  IEEE (IBM/XT,
C    SUN, etc.)  (D.P.)      2        -1022        1024    2.22D-16
C  IBM 3033      (D.P.)     16          -65          63    2.22D-16
C  VAX           (S.P.)      2         -128         127    5.96E-8
C  VAX D-Format  (D.P.)      2         -128         127    1.39D-17
C  VAX G-Format  (D.P.)      2        -1024        1023    1.11D-16
C
C
C                         SQXMIN       XINF        XMIN      XMAX
C
C CRAY-1        (S.P.)  6.77E-1234  5.45E+2465  4.59E-2467 5674.858
C Cyber 180/855
C   under NOS   (S.P.)  1.77E-147   1.26E+322   3.14E-294   672.788
C IEEE (IBM/XT,
C   SUN, etc.)  (S.P.)  1.08E-19    3.40E+38    1.18E-38     85.337
C IEEE (IBM/XT,
C   SUN, etc.)  (D.P.)  1.49D-154   1.79D+308   2.23D-308   705.342
C IBM 3033      (D.P.)  7.35D-40    7.23D+75    5.40D-79    177.852
C VAX           (S.P.)  5.42E-20    1.70E+38    2.94E-39     86.715
C VAX D-Format  (D.P.)  5.42D-20    1.70D+38    2.94D-39     86.715
C VAX G-Format  (D.P.)  7.46D-155   8.98D+307   5.57D-309   706.728
C
C*******************************************************************
C*******************************************************************
C
C Error returns
C
C  In case of an error, NCALC .NE. NB, and not all K's are
C  calculated to the desired accuracy.
C
C  NCALC .LT. -1:  An argument is out of range. For example,
C       NB .LE. 0, IZE is not 1 or 2, or IZE=1 and ABS(X) .GE.
C       XMAX.  In this case, the B-vector is not calculated,
C       and NCALC is set to MIN0(NB,0)-2  so that NCALC .NE. NB.
C  NCALC = -1:  Either  K(ALPHA,X) .GE. XINF  or
C       K(ALPHA+NB-1,X)/K(ALPHA+NB-2,X) .GE. XINF.  In this case,
C       the B-vector is not calculated.  Note that again 
C       NCALC .NE. NB.
C
C  0 .LT. NCALC .LT. NB: Not all requested function values could
C       be calculated accurately.  BK(I) contains correct function
C       values for I .LE. NCALC, and contains the ratios
C       K(ALPHA+I-1,X)/K(ALPHA+I-2,X) for the rest of the array.
C
C
C Intrinsic functions required are:
C
C     ABS, AINT, EXP, INT, LOG, MAX, MIN, SINH, SQRT
C
C
C Acknowledgement
C
C  This program is based on a program written by J. B. Campbell
C  (2) that computes values of the Bessel functions K of real
C  argument and real order.  Modifications include the addition
C  of non-scaled functions, parameterization of machine
C  dependencies, and the use of more accurate approximations
C  for SINH and SIN.
C
C References: "On Temme's Algorithm for the Modified Bessel
C              Functions of the Third Kind," Campbell, J. B.,
C              TOMS 6(4), Dec. 1980, pp. 581-586.
C
C             "A FORTRAN IV Subroutine for the Modified Bessel
C              Functions of the Third Kind of Real Order and Real
C              Argument," Campbell, J. B., Report NRC/ERB-925,
C              National Research Council, Canada.
C
C  Latest modification: May 30, 1989
C
C  Modified by: W. J. Cody and L. Stoltz
C               Applied Mathematics Division
C               Argonne National Laboratory
C               Argonne, IL  60439
C
C-------------------------------------------------------------------
      INTEGER I,IEND,ITEMP,IZE,J,K,M,MPLUS1,NB,NCALC
CS    REAL
      DOUBLE PRECISION  
     1    A,ALPHA,BLPHA,BK,BK1,BK2,C,D,DM,D1,D2,D3,ENU,EPS,ESTF,ESTM,
     2    EX,FOUR,F0,F1,F2,HALF,ONE,P,P0,Q,Q0,R,RATIO,S,SQXMIN,T,TINYX,
     3    TWO,TWONU,TWOX,T1,T2,WMINF,X,XINF,XMAX,XMIN,X2BY4,ZERO
      DIMENSION BK(1),P(8),Q(7),R(5),S(4),T(6),ESTM(6),ESTF(7)
C---------------------------------------------------------------------
C  Mathematical constants
C    A = LOG(2.D0) - Euler's constant
C    D = SQRT(2.D0/PI)
C---------------------------------------------------------------------
CS    DATA HALF,ONE,TWO,ZERO/0.5E0,1.0E0,2.0E0,0.0E0/
CS    DATA FOUR,TINYX/4.0E0,1.0E-10/
CS    DATA A/ 0.11593151565841244881E0/,D/0.797884560802865364E0/
      DATA HALF,ONE,TWO,ZERO/0.5D0,1.0D0,2.0D0,0.0D0/
      DATA FOUR,TINYX/4.0D0,1.0D-10/
      DATA A/ 0.11593151565841244881D0/,D/0.797884560802865364D0/
C---------------------------------------------------------------------
C  Machine dependent parameters
C---------------------------------------------------------------------
CS    DATA EPS/1.19E-7/,SQXMIN/1.08E-19/,XINF/3.40E+38/
CS    DATA XMIN/1.18E-38/,XMAX/85.337E0/
      DATA EPS/2.22D-16/,SQXMIN/1.49D-154/,XINF/1.79D+308/
      DATA XMIN/2.23D-308/,XMAX/705.342D0/
C---------------------------------------------------------------------
C  P, Q - Approximation for LOG(GAMMA(1+ALPHA))/ALPHA
C                                         + Euler's constant
C         Coefficients converted from hex to decimal and modified
C         by W. J. Cody, 2/26/82
C  R, S - Approximation for (1-ALPHA*PI/SIN(ALPHA*PI))/(2.D0*ALPHA)
C  T    - Approximation for SINH(Y)/Y
C---------------------------------------------------------------------
CS    DATA P/ 0.805629875690432845E00,    0.204045500205365151E02,
CS   1        0.157705605106676174E03,    0.536671116469207504E03,
CS   2        0.900382759291288778E03,    0.730923886650660393E03,
CS   3        0.229299301509425145E03,    0.822467033424113231E00/
CS    DATA Q/ 0.294601986247850434E02,    0.277577868510221208E03,
CS   1        0.120670325591027438E04,    0.276291444159791519E04,
CS   2        0.344374050506564618E04,    0.221063190113378647E04,
CS   3        0.572267338359892221E03/
CS    DATA R/-0.48672575865218401848E+0,  0.13079485869097804016E+2,
CS   1       -0.10196490580880537526E+3,  0.34765409106507813131E+3,
CS   2        0.34958981245219347820E-3/
CS    DATA S/-0.25579105509976461286E+2,  0.21257260432226544008E+3,
CS   1       -0.61069018684944109624E+3,  0.42269668805777760407E+3/
CS    DATA T/ 0.16125990452916363814E-9, 0.25051878502858255354E-7,
CS   1        0.27557319615147964774E-5, 0.19841269840928373686E-3,
CS   2        0.83333333333334751799E-2, 0.16666666666666666446E+0/
CS    DATA ESTM/5.20583E1, 5.7607E0, 2.7782E0, 1.44303E1, 1.853004E2,
CS   1          9.3715E0/
CS    DATA ESTF/4.18341E1, 7.1075E0, 6.4306E0, 4.25110E1, 1.35633E0,
CS   1          8.45096E1, 2.0E1/
      DATA P/ 0.805629875690432845D00,    0.204045500205365151D02,
     1        0.157705605106676174D03,    0.536671116469207504D03,
     2        0.900382759291288778D03,    0.730923886650660393D03,
     3        0.229299301509425145D03,    0.822467033424113231D00/
      DATA Q/ 0.294601986247850434D02,    0.277577868510221208D03,
     1        0.120670325591027438D04,    0.276291444159791519D04,
     2        0.344374050506564618D04,    0.221063190113378647D04,
     3        0.572267338359892221D03/
      DATA R/-0.48672575865218401848D+0,  0.13079485869097804016D+2,
     1       -0.10196490580880537526D+3,  0.34765409106507813131D+3,
     2        0.34958981245219347820D-3/
      DATA S/-0.25579105509976461286D+2,  0.21257260432226544008D+3,
     1       -0.61069018684944109624D+3,  0.42269668805777760407D+3/
      DATA T/ 0.16125990452916363814D-9, 0.25051878502858255354D-7,
     1        0.27557319615147964774D-5, 0.19841269840928373686D-3,
     2        0.83333333333334751799D-2, 0.16666666666666666446D+0/
      DATA ESTM/5.20583D1, 5.7607D0, 2.7782D0, 1.44303D1, 1.853004D2,
     1          9.3715D0/
      DATA ESTF/4.18341D1, 7.1075D0, 6.4306D0, 4.25110D1, 1.35633D0,
     1          8.45096D1, 2.0D1/
C---------------------------------------------------------------------
      EX = X
      ENU = ALPHA
      NCALC = MIN(NB,0)-2
      IF ((NB .GT. 0) .AND. ((ENU .GE. ZERO) .AND. (ENU .LT. ONE))
     1     .AND. ((IZE .GE. 1) .AND. (IZE .LE. 2)) .AND.
     2     ((IZE .NE. 1) .OR. (EX .LE. XMAX)) .AND.
     3     (EX .GT. ZERO))  THEN
            K = 0
            IF (ENU .LT. SQXMIN) ENU = ZERO
            IF (ENU .GT. HALF) THEN
                  K = 1
                  ENU = ENU - ONE
            END IF
            TWONU = ENU+ENU
            IEND = NB+K-1
            C = ENU*ENU
            D3 = -C
            IF (EX .LE. ONE) THEN
C---------------------------------------------------------------------
C  Calculation of P0 = GAMMA(1+ALPHA) * (2/X)**ALPHA
C                 Q0 = GAMMA(1-ALPHA) * (X/2)**ALPHA
C---------------------------------------------------------------------
                  D1 = ZERO
                  D2 = P(1)
                  T1 = ONE
                  T2 = Q(1)
                  DO 10 I = 2,7,2
                     D1 = C*D1+P(I)
                     D2 = C*D2+P(I+1)
                     T1 = C*T1+Q(I)
                     T2 = C*T2+Q(I+1)
   10             CONTINUE
                  D1 = ENU*D1
                  T1 = ENU*T1
                  F1 = LOG(EX)
                  F0 = A+ENU*(P(8)-ENU*(D1+D2)/(T1+T2))-F1
                  Q0 = EXP(-ENU*(A-ENU*(P(8)+ENU*(D1-D2)/(T1-T2))-F1))
                  F1 = ENU*F0
                  P0 = EXP(F1)
C---------------------------------------------------------------------
C  Calculation of F0 = 
C---------------------------------------------------------------------
                  D1 = R(5)
                  T1 = ONE
                  DO 20 I = 1,4
                     D1 = C*D1+R(I)
                     T1 = C*T1+S(I)
   20             CONTINUE
                  IF (ABS(F1) .LE. HALF) THEN
                        F1 = F1*F1
                        D2 = ZERO
                        DO 30 I = 1,6
                           D2 = F1*D2+T(I)
   30                   CONTINUE
                        D2 = F0+F0*F1*D2
                     ELSE
                        D2 = SINH(F1)/ENU
                  END IF
                  F0 = D2-ENU*D1/(T1*P0)
                  IF (EX .LE. TINYX) THEN
C--------------------------------------------------------------------
C  X.LE.1.0E-10
C  Calculation of K(ALPHA,X) and X*K(ALPHA+1,X)/K(ALPHA,X)
C--------------------------------------------------------------------
                        BK(1) = F0+EX*F0
                        IF (IZE .EQ. 1) BK(1) = BK(1)-EX*BK(1)
                        RATIO = P0/F0
                        C = EX*XINF
                        IF (K .NE. 0) THEN
C--------------------------------------------------------------------
C  Calculation of K(ALPHA,X) and X*K(ALPHA+1,X)/K(ALPHA,X),
C  ALPHA .GE. 1/2
C--------------------------------------------------------------------
                              NCALC = -1
                              IF (BK(1) .GE. C/RATIO) GO TO 500
                              BK(1) = RATIO*BK(1)/EX
                              TWONU = TWONU+TWO
                              RATIO = TWONU
                        END IF
                        NCALC = 1
                        IF (NB .EQ. 1) GO TO 500
C--------------------------------------------------------------------
C  Calculate  K(ALPHA+L,X)/K(ALPHA+L-1,X),  L  =  1, 2, ... , NB-1
C--------------------------------------------------------------------
                        NCALC = -1
                        DO 80 I = 2,NB
                           IF (RATIO .GE. C) GO TO 500
                           BK(I) = RATIO/EX
                           TWONU = TWONU+TWO
                           RATIO = TWONU
   80                   CONTINUE
                        NCALC = 1
                        GO TO 420
                     ELSE
C--------------------------------------------------------------------
C  1.0E-10 .LT. X .LE. 1.0
C--------------------------------------------------------------------
                        C = ONE
                        X2BY4 = EX*EX/FOUR
                        P0 = HALF*P0
                        Q0 = HALF*Q0
                        D1 = -ONE
                        D2 = ZERO
                        BK1 = ZERO
                        BK2 = ZERO
                        F1 = F0
                        F2 = P0
  100                   D1 = D1+TWO
                        D2 = D2+ONE
                        D3 = D1+D3
                        C = X2BY4*C/D2
                        F0 = (D2*F0+P0+Q0)/D3
                        P0 = P0/(D2-ENU)
                        Q0 = Q0/(D2+ENU)
                        T1 = C*F0
                        T2 = C*(P0-D2*F0)
                        BK1 = BK1+T1
                        BK2 = BK2+T2
                        IF ((ABS(T1/(F1+BK1)) .GT. EPS) .OR.
     1                     (ABS(T2/(F2+BK2)) .GT. EPS))  GO TO 100
                        BK1 = F1+BK1
                        BK2 = TWO*(F2+BK2)/EX
                        IF (IZE .EQ. 2) THEN
                              D1 = EXP(EX)
                              BK1 = BK1*D1
                              BK2 = BK2*D1
                        END IF
                        WMINF = ESTF(1)*EX+ESTF(2)
                  END IF
               ELSE IF (EPS*EX .GT. ONE) THEN
C--------------------------------------------------------------------
C  X .GT. ONE/EPS
C--------------------------------------------------------------------
                  NCALC = NB
                  BK1 = ONE / (D*SQRT(EX))
                  DO 110 I = 1, NB
                     BK(I) = BK1
  110             CONTINUE
                  GO TO 500
               ELSE
C--------------------------------------------------------------------
C  X .GT. 1.0
C--------------------------------------------------------------------
                  TWOX = EX+EX
                  BLPHA = ZERO
                  RATIO = ZERO
                  IF (EX .LE. FOUR) THEN
C--------------------------------------------------------------------
C  Calculation of K(ALPHA+1,X)/K(ALPHA,X),  1.0 .LE. X .LE. 4.0
C--------------------------------------------------------------------
                        D2 = AINT(ESTM(1)/EX+ESTM(2))
                        M = INT(D2)
                        D1 = D2+D2
                        D2 = D2-HALF
                        D2 = D2*D2
                        DO 120 I = 2,M
                           D1 = D1-TWO
                           D2 = D2-D1
                           RATIO = (D3+D2)/(TWOX+D1-RATIO)
  120                   CONTINUE
C--------------------------------------------------------------------
C  Calculation of I(|ALPHA|,X) and I(|ALPHA|+1,X) by backward
C    recurrence and K(ALPHA,X) from the wronskian
C--------------------------------------------------------------------
                        D2 = AINT(ESTM(3)*EX+ESTM(4))
                        M = INT(D2)
                        C = ABS(ENU)
                        D3 = C+C
                        D1 = D3-ONE
                        F1 = XMIN
                        F0 = (TWO*(C+D2)/EX+HALF*EX/(C+D2+ONE))*XMIN
                        DO 130 I = 3,M
                           D2 = D2-ONE
                           F2 = (D3+D2+D2)*F0
                           BLPHA = (ONE+D1/D2)*(F2+BLPHA)
                           F2 = F2/EX+F1
                           F1 = F0
                           F0 = F2
  130                   CONTINUE
                        F1 = (D3+TWO)*F0/EX+F1
                        D1 = ZERO
                        T1 = ONE
                        DO 140 I = 1,7
                           D1 = C*D1+P(I)
                           T1 = C*T1+Q(I)
  140                   CONTINUE
                        P0 = EXP(C*(A+C*(P(8)-C*D1/T1)-LOG(EX)))/EX
                        F2 = (C+HALF-RATIO)*F1/EX
                        BK1 = P0+(D3*F0-F2+F0+BLPHA)/(F2+F1+F0)*P0
                        IF (IZE .EQ. 1) BK1 = BK1*EXP(-EX)
                        WMINF = ESTF(3)*EX+ESTF(4)
                     ELSE
C--------------------------------------------------------------------
C  Calculation of K(ALPHA,X) and K(ALPHA+1,X)/K(ALPHA,X), by backward
C  recurrence, for  X .GT. 4.0
C--------------------------------------------------------------------
                        DM = AINT(ESTM(5)/EX+ESTM(6))
                        M = INT(DM)
                        D2 = DM-HALF
                        D2 = D2*D2
                        D1 = DM+DM
                        DO 160 I = 2,M
                           DM = DM-ONE
                           D1 = D1-TWO
                           D2 = D2-D1
                           RATIO = (D3+D2)/(TWOX+D1-RATIO)
                           BLPHA = (RATIO+RATIO*BLPHA)/DM
  160                   CONTINUE
                        BK1 = ONE/((D+D*BLPHA)*SQRT(EX))
                        IF (IZE .EQ. 1) BK1 = BK1*EXP(-EX)
                        WMINF = ESTF(5)*(EX-ABS(EX-ESTF(7)))+ESTF(6)
                  END IF
C--------------------------------------------------------------------
C  Calculation of K(ALPHA+1,X) from K(ALPHA,X) and
C    K(ALPHA+1,X)/K(ALPHA,X)
C--------------------------------------------------------------------
                  BK2 = BK1+BK1*(ENU+HALF-RATIO)/EX
            END IF
C--------------------------------------------------------------------
C  Calculation of 'NCALC', K(ALPHA+I,X), I  =  0, 1, ... , NCALC-1,
C  K(ALPHA+I,X)/K(ALPHA+I-1,X), I  =  NCALC, NCALC+1, ... , NB-1
C--------------------------------------------------------------------
            NCALC = NB
            BK(1) = BK1
            IF (IEND .EQ. 0) GO TO 500
            J = 2-K
            IF (J .GT. 0) BK(J) = BK2
            IF (IEND .EQ. 1) GO TO 500
            M = MIN(INT(WMINF-ENU),IEND)
            DO 190 I = 2,M
               T1 = BK1
               BK1 = BK2
               TWONU = TWONU+TWO
               IF (EX .LT. ONE) THEN
                     IF (BK1 .GE. (XINF/TWONU)*EX) GO TO 195
                     GO TO 187
                  ELSE 
                     IF (BK1/EX .GE. XINF/TWONU) GO TO 195
               END IF
  187          CONTINUE
               BK2 = TWONU/EX*BK1+T1
               ITEMP = I
               J = J+1
               IF (J .GT. 0) BK(J) = BK2
  190       CONTINUE
  195       M = ITEMP
            IF (M .EQ. IEND) GO TO 500
            RATIO = BK2/BK1
            MPLUS1 = M+1
            NCALC = -1
            DO 410 I = MPLUS1,IEND
               TWONU = TWONU+TWO
               RATIO = TWONU/EX+ONE/RATIO
               J = J+1
               IF (J .GT. 1) THEN
                     BK(J) = RATIO
                  ELSE
                     IF (BK2 .GE. XINF/RATIO) GO TO 500
                     BK2 = RATIO*BK2
               END IF
  410       CONTINUE
            NCALC = MAX(MPLUS1-K,1)
            IF (NCALC .EQ. 1) BK(1) = BK2
            IF (NB .EQ. 1) GO TO 500
  420       J = NCALC+1
            DO 430 I = J,NB
               IF (BK(NCALC) .GE. XINF/BK(I)) GO TO 500
               BK(I) = BK(NCALC)*BK(I)
               NCALC = I
  430       CONTINUE
      END IF
  500 RETURN
C---------- Last line of RKBESL ----------
      END

c
      SUBROUTINE PPND16 (P, IFAULT)
cf2py intent(inout) p
cf2py intent(out) ifault
C
C      ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3
C
C      Produces the normal deviate Z corresponding to a given lower
C      tail area of P; Z is accurate to about 1 part in 10**16.
C
C      The hash sums below are the sums of the mantissas of the
C      coefficients.   They are included for use in checking
C      transcription.
C
      DOUBLE PRECISION P
      DOUBLE PRECISION ZERO, ONE, HALF, SPLIT1, SPLIT2, CONST1,
     *           CONST2, A0, A1, A2, A3, A4, A5, A6, A7, 
     *           B1, B2, B3, B4, B5, B6, B7,
     *           C0, C1, C2, C3, C4, C5, C6, C7,
     *           D1, D2, D3, D4, D5, D6, D7, 
     *           E0, E1, E2, E3, E4, E5, E6, E7, 
     *           F1, F2, F3, F4, F5, F6, F7, Q, R
      PARAMETER (ZERO = 0.D0, ONE = 1.D0, HALF = 0.5D0,
     *            SPLIT1 = 0.425D0, SPLIT2 = 5.D0,
     *            CONST1 = 0.180625D0, CONST2 = 1.6D0)
C
C      Coefficients for P close to 0.5
C
      PARAMETER (A0 = 3.3871328727963666080D0,
     *           A1 = 1.3314166789178437745D+2,
     *           A2 = 1.9715909503065514427D+3,
     *           A3 = 1.3731693765509461125D+4,
     *           A4 = 4.5921953931549871457D+4,
     *           A5 = 6.7265770927008700853D+4,
     *           A6 = 3.3430575583588128105D+4,
     *           A7 = 2.5090809287301226727D+3,
     *           B1 = 4.2313330701600911252D+1,
     *           B2 = 6.8718700749205790830D+2,
     *           B3 = 5.3941960214247511077D+3,
     *           B4 = 2.1213794301586595867D+4,
     *           B5 = 3.9307895800092710610D+4,
     *           B6 = 2.8729085735721942674D+4,
     *           B7 = 5.2264952788528545610D+3)
C      HASH SUM AB    55.88319 28806 14901 4439
C
C      Coefficients for P not close to 0, 0.5 or 1.
C
      PARAMETER (C0 = 1.42343711074968357734D0,
     *          C1 = 4.63033784615654529590D0,
     *          C2 = 5.76949722146069140550D0,
     *          C3 = 3.64784832476320460504D0,
     *          C4 = 1.27045825245236838258D0,
     *          C5 = 2.41780725177450611770D-1,
     *          C6=2.27238449892691845833D-2,
     *          C7 = 7.74545014278341407640D-4,
     *          D1 = 2.05319162663775882187D0,
     *          D2 = 1.67638483018380384940D0,
     *          D3 = 6.89767334985100004550D-1,
     *          D4 = 1.48103976427480074590D-1,
     *          D5 = 1.51986665636164571966D-2,
     *          D6 = 5.47593808499534494600D-4,
     *          D7 = 1.05075007164441684324D-9)
C      HASH SUM       49.33206 50330 16102 89036
C
C      Coefficients for P near 0 or 1.
C
      INTEGER IFAULT
      PARAMETER (E0 = 6.65790464350110377720D0,
     *          E1 = 5.46378491116411436990D0,
     *          E2 = 1.78482653991729133580D0,
     *          E3 = 2.96560571828504891230D-1,
     *          E4 = 2.65321895265761230930D-2,
     *          E5 = 1.24266094738807843860D-3,
     *          E6 = 2.71155556874348757815D-5,
     *          E7 = 2.01033439929228813265D-7,
     *          F1 = 5.99832206555887937690D-1,
     *          F2 = 1.36929880922735805310D-1,
     *          F3 = 1.48753612908506148525D-2,
     *          F4 = 7.86869131145613259100D-4,
     *          F5 = 1.84631831751005468180D-5,
     *          F6 = 1.42151175831644588870D-7,
     *          F7 = 2.04426310338993978564D-15)
C      HASH SUM EF    47.52583 31754 92896 71629
C
      IFAULT = 0
      Q = P - HALF
      IF (ABS(Q) .LE. SPLIT1) THEN
        R = CONST1 - Q * Q
        P = Q * (((((((A7 * R + A6) * R + A5) * R + A4) * R + A3)
     *             * R + A2) * R + A1) * R + A0) /
     *             (((((((B7 * R + B6) * R + B5) * R + B4) * R + B3)
     *             * R + B2) * R + B1) * R + ONE)
        RETURN
      ELSE
        IF (Q .LT. ZERO) THEN
          R = P
        ELSE
          R = ONE - P
        END IF
        IF (R .LE. ZERO) THEN
          IFAULT = 1
          P = ZERO
          RETURN
        END IF
        R = SQRT(-LOG(R))
        IF (R .LE. SPLIT2) THEN
          R = R - CONST2
          P = (((((((C7 * R + C6) * R + C5) * R + C4) * R + C3)
     *              * R + C2) * R + C1) * R + C0) /
     *              (((((((D7 * R + D6) * R + D5) * R + D4) * R + D3)
     *              * R + D2) * R + D1) * R + ONE)
        ELSE
          R = R - SPLIT2
          P = (((((((E7 * R + E6) * R + E5) * R + E4) * R + E3)
     *             * R + E2) * R + E1) * R + E0) /
     *             (((((((F7 * R + F6) * R + F5) * R + F4) * R + F3)
     *             * R + F2) * R + F1) * R + ONE)
        END IF
        IF (Q .LT. ZERO) P = - P
        RETURN
      END IF
      END


      SUBROUTINE CALERF(ARG,RESULT,JINT)
C------------------------------------------------------------------
C
C This packet evaluates  erf(x),  erfc(x),  and  exp(x*x)*erfc(x)
C   for a real argument  x.  It contains three FUNCTION type
C   subprograms: ERF, ERFC, and ERFCX (or DERF, DERFC, and DERFCX),
C   and one SUBROUTINE type subprogram, CALERF.  The calling
C   statements for the primary entries are:
C
C                   Y=ERF(X)     (or   Y=DERF(X)),
C
C                   Y=ERFC(X)    (or   Y=DERFC(X)),
C   and
C                   Y=ERFCX(X)   (or   Y=DERFCX(X)).
C
C   The routine  CALERF  is intended for internal packet use only,
C   all computations within the packet being concentrated in this
C   routine.  The function subprograms invoke  CALERF  with the
C   statement
C
C          CALL CALERF(ARG,RESULT,JINT)
C
C   where the parameter usage is as follows
C
C      Function                     Parameters for CALERF
C       call              ARG                  Result          JINT
C
C     ERF(ARG)      ANY REAL ARGUMENT         ERF(ARG)          0
C     ERFC(ARG)     ABS(ARG) .LT. XBIG        ERFC(ARG)         1
C     ERFCX(ARG)    XNEG .LT. ARG .LT. XMAX   ERFCX(ARG)        2
C
C   The main computation evaluates near-minimax approximations
C   from "Rational Chebyshev approximations for the error function"
C   by W. J. Cody, Math. Comp., 1969, PP. 631-638.  This
C   transportable program uses rational functions that theoretically
C   approximate  erf(x)  and  erfc(x)  to at least 18 significant
C   decimal digits.  The accuracy achieved depends on the arithmetic
C   system, the compiler, the intrinsic functions, and proper
C   selection of the machine-dependent constants.
C
C*******************************************************************
C*******************************************************************
C
C Explanation of machine-dependent constants
C
C   XMIN   = the smallest positive floating-point number.
C   XINF   = the largest positive finite floating-point number.
C   XNEG   = the largest negative argument acceptable to ERFCX;
C            the negative of the solution to the equation
C            2*exp(x*x) = XINF.
C   XSMALL = argument below which erf(x) may be represented by
C            2*x/sqrt(pi)  and above which  x*x  will not underflow.
C            A conservative value is the largest machine number X
C            such that   1.0 + X = 1.0   to machine precision.
C   XBIG   = largest argument acceptable to ERFC;  solution to
C            the equation:  W(x) * (1-0.5/x**2) = XMIN,  where
C            W(x) = exp(-x*x)/[x*sqrt(pi)].
C   XHUGE  = argument above which  1.0 - 1/(2*x*x) = 1.0  to
C            machine precision.  A conservative value is
C            1/[2*sqrt(XSMALL)]
C   XMAX   = largest acceptable argument to ERFCX; the minimum
C            of XINF and 1/[sqrt(pi)*XMIN].
C
C   Approximate values for some important machines are:
C
C                          XMIN       XINF        XNEG     XSMALL
C
C    C 7600      (S.P.)  3.13E-294   1.26E+322   -27.220  7.11E-15
C  CRAY-1        (S.P.)  4.58E-2467  5.45E+2465  -75.345  7.11E-15
C  IEEE (IBM/XT,
C    SUN, etc.)  (S.P.)  1.18E-38    3.40E+38     -9.382  5.96E-8
C  IEEE (IBM/XT,
C    SUN, etc.)  (D.P.)  2.23D-308   1.79D+308   -26.628  1.11D-16
C  IBM 195       (D.P.)  5.40D-79    7.23E+75    -13.190  1.39D-17
C  UNIVAC 1108   (D.P.)  2.78D-309   8.98D+307   -26.615  1.73D-18
C  VAX D-Format  (D.P.)  2.94D-39    1.70D+38     -9.345  1.39D-17
C  VAX G-Format  (D.P.)  5.56D-309   8.98D+307   -26.615  1.11D-16
C
C
C                          XBIG       XHUGE       XMAX
C
C    C 7600      (S.P.)  25.922      8.39E+6     1.80X+293
C  CRAY-1        (S.P.)  75.326      8.39E+6     5.45E+2465
C  IEEE (IBM/XT,
C    SUN, etc.)  (S.P.)   9.194      2.90E+3     4.79E+37
C  IEEE (IBM/XT,
C    SUN, etc.)  (D.P.)  26.543      6.71D+7     2.53D+307
C  IBM 195       (D.P.)  13.306      1.90D+8     7.23E+75
C  UNIVAC 1108   (D.P.)  26.582      5.37D+8     8.98D+307
C  VAX D-Format  (D.P.)   9.269      1.90D+8     1.70D+38
C  VAX G-Format  (D.P.)  26.569      6.71D+7     8.98D+307
C
C*******************************************************************
C*******************************************************************
C
C Error returns
C
C  The program returns  ERFC = 0      for  ARG .GE. XBIG;
C
C                       ERFCX = XINF  for  ARG .LT. XNEG;
C      and
C                       ERFCX = 0     for  ARG .GE. XMAX.
C
C
C Intrinsic functions required are:
C
C     ABS, AINT, EXP
C
C
C  Author: W. J. Cody
C          Mathematics and Computer Science Division
C          Argonne National Laboratory
C          Argonne, IL 60439
C
C  Latest modification: March 19, 1990
C
C------------------------------------------------------------------
      INTEGER I,JINT
CS    REAL
      DOUBLE PRECISION
     1     A,ARG,B,C,D,DEL,FOUR,HALF,P,ONE,Q,RESULT,SIXTEN,SQRPI,
     2     TWO,THRESH,X,XBIG,XDEN,XHUGE,XINF,XMAX,XNEG,XNUM,XSMALL,
     3     Y,YSQ,ZERO
      DIMENSION A(5),B(4),C(9),D(8),P(6),Q(5)
C------------------------------------------------------------------
C  Mathematical constants
C------------------------------------------------------------------
CS    DATA FOUR,ONE,HALF,TWO,ZERO/4.0E0,1.0E0,0.5E0,2.0E0,0.0E0/,
CS   1     SQRPI/5.6418958354775628695E-1/,THRESH/0.46875E0/,
CS   2     SIXTEN/16.0E0/
      DATA FOUR,ONE,HALF,TWO,ZERO/4.0D0,1.0D0,0.5D0,2.0D0,0.0D0/,
     1     SQRPI/5.6418958354775628695D-1/,THRESH/0.46875D0/,
     2     SIXTEN/16.0D0/
C------------------------------------------------------------------
C  Machine-dependent constants
C------------------------------------------------------------------
CS    DATA XINF,XNEG,XSMALL/3.40E+38,-9.382E0,5.96E-8/,
CS   1     XBIG,XHUGE,XMAX/9.194E0,2.90E3,4.79E37/
      DATA XINF,XNEG,XSMALL/1.79D308,-26.628D0,1.11D-16/,
     1     XBIG,XHUGE,XMAX/26.543D0,6.71D7,2.53D307/
C------------------------------------------------------------------
C  Coefficients for approximation to  erf  in first interval
C------------------------------------------------------------------
CS    DATA A/3.16112374387056560E00,1.13864154151050156E02,
CS   1       3.77485237685302021E02,3.20937758913846947E03,
CS   2       1.85777706184603153E-1/
CS    DATA B/2.36012909523441209E01,2.44024637934444173E02,
CS   1       1.28261652607737228E03,2.84423683343917062E03/
      DATA A/3.16112374387056560D00,1.13864154151050156D02,
     1       3.77485237685302021D02,3.20937758913846947D03,
     2       1.85777706184603153D-1/
      DATA B/2.36012909523441209D01,2.44024637934444173D02,
     1       1.28261652607737228D03,2.84423683343917062D03/
C------------------------------------------------------------------
C  Coefficients for approximation to  erfc  in second interval
C------------------------------------------------------------------
CS    DATA C/5.64188496988670089E-1,8.88314979438837594E0,
CS   1       6.61191906371416295E01,2.98635138197400131E02,
CS   2       8.81952221241769090E02,1.71204761263407058E03,
CS   3       2.05107837782607147E03,1.23033935479799725E03,
CS   4       2.15311535474403846E-8/
CS    DATA D/1.57449261107098347E01,1.17693950891312499E02,
CS   1       5.37181101862009858E02,1.62138957456669019E03,
CS   2       3.29079923573345963E03,4.36261909014324716E03,
CS   3       3.43936767414372164E03,1.23033935480374942E03/
      DATA C/5.64188496988670089D-1,8.88314979438837594D0,
     1       6.61191906371416295D01,2.98635138197400131D02,
     2       8.81952221241769090D02,1.71204761263407058D03,
     3       2.05107837782607147D03,1.23033935479799725D03,
     4       2.15311535474403846D-8/
      DATA D/1.57449261107098347D01,1.17693950891312499D02,
     1       5.37181101862009858D02,1.62138957456669019D03,
     2       3.29079923573345963D03,4.36261909014324716D03,
     3       3.43936767414372164D03,1.23033935480374942D03/
C------------------------------------------------------------------
C  Coefficients for approximation to  erfc  in third interval
C------------------------------------------------------------------
CS    DATA P/3.05326634961232344E-1,3.60344899949804439E-1,
CS   1       1.25781726111229246E-1,1.60837851487422766E-2,
CS   2       6.58749161529837803E-4,1.63153871373020978E-2/
CS    DATA Q/2.56852019228982242E00,1.87295284992346047E00,
CS   1       5.27905102951428412E-1,6.05183413124413191E-2,
CS   2       2.33520497626869185E-3/
      DATA P/3.05326634961232344D-1,3.60344899949804439D-1,
     1       1.25781726111229246D-1,1.60837851487422766D-2,
     2       6.58749161529837803D-4,1.63153871373020978D-2/
      DATA Q/2.56852019228982242D00,1.87295284992346047D00,
     1       5.27905102951428412D-1,6.05183413124413191D-2,
     2       2.33520497626869185D-3/
C------------------------------------------------------------------
      X = ARG
      Y = ABS(X)
      IF (Y .LE. THRESH) THEN
C------------------------------------------------------------------
C  Evaluate  erf  for  |X| <= 0.46875
C------------------------------------------------------------------
            YSQ = ZERO
            IF (Y .GT. XSMALL) YSQ = Y * Y
            XNUM = A(5)*YSQ
            XDEN = YSQ
            DO 20 I = 1, 3
               XNUM = (XNUM + A(I)) * YSQ
               XDEN = (XDEN + B(I)) * YSQ
   20       CONTINUE
            RESULT = X * (XNUM + A(4)) / (XDEN + B(4))
            IF (JINT .NE. 0) RESULT = ONE - RESULT
            IF (JINT .EQ. 2) RESULT = EXP(YSQ) * RESULT
            GO TO 800
C------------------------------------------------------------------
C  Evaluate  erfc  for 0.46875 <= |X| <= 4.0
C------------------------------------------------------------------
         ELSE IF (Y .LE. FOUR) THEN
            XNUM = C(9)*Y
            XDEN = Y
            DO 120 I = 1, 7
               XNUM = (XNUM + C(I)) * Y
               XDEN = (XDEN + D(I)) * Y
  120       CONTINUE
            RESULT = (XNUM + C(8)) / (XDEN + D(8))
            IF (JINT .NE. 2) THEN
               YSQ = AINT(Y*SIXTEN)/SIXTEN
               DEL = (Y-YSQ)*(Y+YSQ)
               RESULT = EXP(-YSQ*YSQ) * EXP(-DEL) * RESULT
            END IF
C------------------------------------------------------------------
C  Evaluate  erfc  for |X| > 4.0
C------------------------------------------------------------------
         ELSE
            RESULT = ZERO
            IF (Y .GE. XBIG) THEN
               IF ((JINT .NE. 2) .OR. (Y .GE. XMAX)) GO TO 300
               IF (Y .GE. XHUGE) THEN
                  RESULT = SQRPI / Y
                  GO TO 300
               END IF
            END IF
            YSQ = ONE / (Y * Y)
            XNUM = P(6)*YSQ
            XDEN = YSQ
            DO 240 I = 1, 4
               XNUM = (XNUM + P(I)) * YSQ
               XDEN = (XDEN + Q(I)) * YSQ
  240       CONTINUE
            RESULT = YSQ *(XNUM + P(5)) / (XDEN + Q(5))
            RESULT = (SQRPI -  RESULT) / Y
            IF (JINT .NE. 2) THEN
               YSQ = AINT(Y*SIXTEN)/SIXTEN
               DEL = (Y-YSQ)*(Y+YSQ)
               RESULT = EXP(-YSQ*YSQ) * EXP(-DEL) * RESULT
            END IF
      END IF
C------------------------------------------------------------------
C  Fix up for negative argument, erf, etc.
C------------------------------------------------------------------
  300 IF (JINT .EQ. 0) THEN
            RESULT = (HALF - RESULT) + HALF
            IF (X .LT. ZERO) RESULT = -RESULT
         ELSE IF (JINT .EQ. 1) THEN
            IF (X .LT. ZERO) RESULT = TWO - RESULT
         ELSE
            IF (X .LT. ZERO) THEN
               IF (X .LT. XNEG) THEN
                     RESULT = XINF
                  ELSE
                     YSQ = AINT(X*SIXTEN)/SIXTEN
                     DEL = (X-YSQ)*(X+YSQ)
                     Y = EXP(YSQ*YSQ) * EXP(DEL)
                     RESULT = (Y+Y) - RESULT
               END IF
            END IF
      END IF
  800 RETURN
C---------- Last card of CALERF ----------
      END
CS    REAL FUNCTION ERF(X)
!       DOUBLE PRECISION FUNCTION DERF(X)
! C--------------------------------------------------------------------
! C
! C This subprogram computes approximate values for erf(x).
! C   (see comments heading CALERF).
! C
! C   Author/date: W. J. Cody, January 8, 1985
! C
! C--------------------------------------------------------------------
!       INTEGER JINT
! CS    REAL             X, RESULT
!       DOUBLE PRECISION X, RESULT
! C------------------------------------------------------------------
! !       if (X<-8.0D0) then
! !           DERF=0.0D0
! !       else if (X>8.0D0) then
! !           DERF=1.0D0
! !       else
!           JINT = 0
!           CALL CALERF(X,RESULT,JINT)
!           DERF = RESULT
! !       end if
!       RETURN
! C---------- Last card of DERF ----------
!       END
!       
!       DOUBLE PRECISION FUNCTION LERFC(X)
!       DOUBLE PRECISION PI, RESULT
!       INTEGER I, N, NFAC, N2FAC
!     
!       DO I=0,4
!         
!       END DO
!       
!       RETURN
!       END
      
CS    REAL FUNCTION ERFC(X)
      DOUBLE PRECISION FUNCTION DERFC(X)
C--------------------------------------------------------------------
C
C This subprogram computes approximate values for erfc(x).
C   (see comments heading CALERF).
C
C   Author/date: W. J. Cody, January 8, 1985
C
C--------------------------------------------------------------------
      INTEGER JINT
CS    REAL             X, RESULT
      DOUBLE PRECISION X, RESULT
C------------------------------------------------------------------
      JINT = 1
      CALL CALERF(X,RESULT,JINT)
CS    ERFC = RESULT
      DERFC = RESULT
      RETURN
C---------- Last card of DERFC ----------
      END
CS    REAL FUNCTION ERFCX(X)
      DOUBLE PRECISION FUNCTION DERFCX(X)
C------------------------------------------------------------------
C
C This subprogram computes approximate values for exp(x*x) * erfc(x).
C   (see comments heading CALERF).
C
C   Author/date: W. J. Cody, March 30, 1987
C
C------------------------------------------------------------------
      INTEGER JINT
CS    REAL             X, RESULT
      DOUBLE PRECISION X, RESULT
C------------------------------------------------------------------
      JINT = 2
      CALL CALERF(X,RESULT,JINT)
CS    ERFCX = RESULT
      DERFCX = RESULT
      RETURN
C---------- Last card of DERFCX ----------
      END
