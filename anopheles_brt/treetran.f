
      SUBROUTINE treetran(s, l, r, v, o, nt, no)
cf2py intent(inplace) o
cf2py intent(hide) nt, no
      DOUBLE PRECISION s(nt), l(nt), r(nt), v(no), o(no)
      INTEGER nt, no, i, j
     
c s: splitpoint
c l: left value
c r: right value
c v: values of variable
c o: output to be overwritten
      do i=1,nt
         do j=1,no
             if (v(j).LT.s(i)) then
                 o(j) = o(j) + l(i)
             else
                o(j) = o(j) + r(i)
             end if
         end do
      end do
      
      RETURN
      END