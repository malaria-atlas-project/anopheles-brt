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

!       SUBROUTINE binomial(x,n,p,nx,nn,np,like)
! 
! c Binomial log-likelihood function     
! 
! c  Updated 17/01/2007. DH. 
! 
! cf2py integer dimension(nx),intent(in) :: x
! cf2py integer dimension(nn),intent(in) :: n
! cf2py double precision dimension(np),intent(in) :: p
! cf2py integer intent(hide),depend(x) :: nx=len(x)
! cf2py integer intent(hide),depend(n),check(nn==1 || nn==len(x)) :: nn=len(n)
! cf2py integer intent(hide),depend(p),check(np==1 || np==len(x)) :: np=len(p)
! cf2py double precision intent(out) :: like      
! cf2py threadsafe
!       IMPLICIT NONE
!       INTEGER nx,nn,np,i
!       DOUBLE PRECISION like, p(np)
!       INTEGER x(nx),n(nn)
!       LOGICAL not_scalar_n,not_scalar_p
!       INTEGER ntmp
!       DOUBLE PRECISION ptmp
!       DOUBLE PRECISION factln
!       DOUBLE PRECISION infinity
!       PARAMETER (infinity = 1.7976931348623157d308)
! 
!       not_scalar_n = (nn .NE. 1)
!       not_scalar_p = (np .NE. 1) 
! 
!       ntmp = n(1)
!       ptmp = p(1)
! 
!       like = 0.0
!       do i=1,nx
!         if (not_scalar_n) ntmp = n(i)
!         if (not_scalar_p) ptmp = p(i)
!         
!         if ((x(i) .LT. 0) .OR. (ntmp .LT. 0) .OR. (x(i) .GT. ntmp)) then
!           like = -infinity
!           RETURN
!         endif
!         
!         if ((ptmp .LE. 0.0D0) .OR. (ptmp .GE. 1.0D0)) then
! !         if p = 0, number of successes must be 0
!           if (ptmp .EQ. 0.0D0) then
!             if (x(i) .GT. 0.0D0) then
!                 like = -infinity
!                 RETURN
! !                 else like = like + 0
!             end if
!           else if (ptmp .EQ. 1.0D0) then
! !           if p = 1, number of successes must be n
!             if (x(i) .LT. ntmp) then
!                 like = -infinity
!                 RETURN
! !                 else like = like + 0
!             end if
!           else
!             like = -infinity
!             RETURN
!           endif
!         else
!             like = like + x(i)*dlog(ptmp) + (ntmp-x(i))*dlog(1.-ptmp)
!             like = like + factln(ntmp)-factln(x(i))-factln(ntmp-x(i)) 
!         end if
!       enddo
!       return
!       END

      SUBROUTINE logsum(z,x,y)
      
! Computes z <- x + (-1)^s * y.

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

      SUBROUTINE ubl(o,n,p)
! Unequal binomial log-probability distribution
cf2py intent(out) o
      INTEGER n, i, j
      DOUBLE PRECISION p(n), o(n+1), last(n)
      DOUBLE PRECISION lp(n), lomp(n)
      
! Log of p and 1-p
      
      do i=1,n
          lp(i) = dlog(p(i))
          lomp(i) = dlog(1.0D0-p(i))
      end do
      
      o(1) = lomp(1)
      o(2) = lp(1)
      
      do i=2,n
          do j=1,i
              last(j)=o(j)
          end do
          o(i+1) = last(i) + lp(i)
          do j=i,2,-1
!               o(j) = dlog(dexp(last(j-1)+lp(i))+dexp(last(j)+lomp(i)))
             CALL logsum(o(j), last(j-1)+lp(i), last(j)+lomp(i))
          end do
          o(1) = last(1) + lomp(i)
      end do

      RETURN
      END

      SUBROUTINE mahal(c,x,y,symm,a,l,s,nx,ny,nd,cmin,cmax)
cf2py intent(hide) nx,ny,nd
cf2py intent(inplace) c
cf2py threadsafe
      DOUBLE PRECISION x(nx,nd), y(ny,nd), s(nd,nd), l(nd)
      DOUBLE PRECISION c(nx,ny), dev(nd), this, a, tdev(nd)
      INTEGER i,j,k,m,nx,ny,nd,cmin,cmax
      LOGICAL symm
      
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

                    c(i,j) = dexp(this)*a*a

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
              
                  c(i,j) = dexp(this)*a*a

              end do
          end if
      end do
      
      RETURN 
      END