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

      SUBROUTINE mahal(c,x,y,symm,a,l,s,nx,ny,nd,cmin,cmax)
cf2py intent(out) d
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