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


      DOUBLE PRECISION FUNCTION factln(n) 
C USES gammln Returns ln(n!). 

      IMPLICIT NONE
      INTEGER n 
      DOUBLE PRECISION a(100),gammln, pass_val 
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)
      
      SAVE a 
C Initialize the table to negative values. 
      DATA a/100*-1./ 
      pass_val = n + 1
      if (n.lt.0) then
c        write (*,*) 'negative factorial in factln' 
        factln=-infinity
        return
      endif
C In range of the table. 
      if (n.le.99) then
C If not already in the table, put it in.
        if (a(n+1).lt.0.) a(n+1)=gammln(pass_val) 
        factln=a(n+1) 
      else 
C Out of range of the table. 
        factln=gammln(pass_val) 
      endif 
      return 
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

      SUBROUTINE ubl(o,n,p)
      
! Unequal binomial log-probability distribution
cf2py intent(out) o
cf2py intent(hide) n
cf2py threadsafe

      IMPLICIT NONE
      INTEGER n, i, j
      DOUBLE PRECISION o(n+1), last(n), p(n)
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
             CALL logsum2(o(j), last(j-1)+lp(i), last(j)+lomp(i))
          end do
          o(1) = last(1) + lomp(i)
      end do

      RETURN
      END
      
      SUBROUTINE bin_ubl(like,npos,n,q,npix,p)

c Binomial log-likelihood function mixed on p as in the anopheline project.

cf2py intent(out) like
cf2py intent(hide) npix
cf2py threadsafe

      INTEGER npos,k,npix
      DOUBLE PRECISION like,q,p(npix),lpk(npix+1),likes(npix+1)
      DOUBLE PRECISION factln, f
      DOUBLE PRECISION infinity
      PARAMETER (infinity = 1.7976931348623157d308)

! Get the log-density of k
      CALL ubl(lpk,npix,p)
      
! The zero term: no pixels have the thing.
      like = 0.0
      if (npos.EQ.0) then
          likes(1)=lpk(1)
      else
        likes(1)=-infinity
      end if
      
! The non-zero terms: Some pixels have it.
      do k=1,npix
          f=q*k/npix
          likes(k+1)=lpk(k+1)+npos*dlog(f)+(n-npos)*dlog(1.0D0-f)
      if ((npos.LT.n).AND.(npos.GT.0)) then
          likes(k+1)=likes(k+1)+factln(n)-factln(npos)-factln(n-npos)
      end if          
      end do
      
!       print *,likes

! Log-sum and normalize
      CALL logsum(likes,npix+1,like)
          
      RETURN
      END

      SUBROUTINE bin_ubls(like,nposs,ns,q,breaks,ps,nobs,npts)

c Calls bin_ubl for multiple observations. 
c Just eliminates Python looping and slicing.      
cf2py intent(out) like
cf2py intent(hide) nobs,npts
cf2py threadsafe
      
      DOUBLE PRECISION like,this_like
      DOUBLE PRECISION q,ps(npts)
      INTEGER nobs,i,ns(nobs),nposs(nobs),breaks(nobs+1)
      INTEGER npix,runtot,npts
      
      runtot = 1
      npix = 0
      like = 0.0D0
      do i=1,nobs
          runtot = runtot+npix
          npix = breaks(i+1)-breaks(i)
          CALL bin_ubl(this_like,nposs(i),ns(i),q,npix,ps(runtot))
          like = like + this_like
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


      SUBROUTINE mod_mahal(c,ds,x,y,symm,a,l,s,cf,nx,ny,nd,cmin,cmax)
cf2py intent(hide) nx,ny,nd
cf2py intent(inplace) c
cf2py threadsafe
      DOUBLE PRECISION x(nx,nd), y(ny,nd), s(nd+1,nd+1), l(nd+1)
      DOUBLE PRECISION c(nx,ny), dev(nd+1), this, a, tdev(nd+1)
      DOUBLE PRECISION ds(nx,ny)
      DOUBLE PRECISION cf
      INTEGER i,j,k,m,nx,ny,nd,cmin,cmax
      LOGICAL symm
      
!       DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
      EXTERNAL DGEMV
      
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
                    
!                     print *,dev(1),tdev(1),dsqrt(this)

                    c(i,j) = (dexp(-dsqrt(this))*(1.0D0-cf)+cf)*a*a

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
              
                  c(i,j) = (dexp(-dsqrt(this))*(1.0D0-cf)+cf)*a*a

              end do
          end if
      end do
      
      RETURN 
      END


