

       program main
       implicit none
       integer :: nonods_l, ndim, nscalar
       integer :: nonods
       parameter(ndim=1,nscalar=1)
       parameter(nonods_l=19,nonods=4)
       real :: x_l(ndim,nonods_l), conc_l(nscalar,nonods_l)
       real :: x(ndim,nonods), conc(nscalar,nonods)
       integer :: i
       
       print *,'me'
       do i=1,nonods
          x(1,i) = real(i)
       end do
       conc=1.0
       conc(1,:)=x(1,:)
       call x_conv_fixed_length(x_l,conc_l, x,conc, nonods_l,nonods,ndim,nscalar)
       print *,'nonods_l,nonods:',nonods_l,nonods
       print *,'x:',x
       print *,'x_l:',x_l
       print *,'conc:'
       print *,'conc:',conc
       print *,'conc_l:',conc_l
       return
       end program main
! 
! 
! 
! 
! from python call...
! x_l,conc_l = x_conv_fixed_length(x,conc,nonods,nonods_l,ndim,nscalar)
       subroutine x_conv_fixed_length(x_l,conc_l, x,conc, nonods_l,nonods,ndim,nscalar)
! ************************************************************************************
! This subroutine calculates x_l from x & conc_l from conc by linearly interpolating 
! from a regular grid
! ************************************************************************************
! nonods = no of nodes in mesh to be interpolated from. 
! nonods_l = no of nodes in mesh to be interpolated too. 
! ndim= no of dimensions e.g. for 3D problems =3. 
! nscalar=no of concentration fields to to be interpolated. 
! conc_l= the concentration field to be ionterpolated too.
! conc= the concentration field to be ionterpolated from.
! x_l= the spatial coordinates to be ionterpolated too.
! x= the spatial coordinates to be ionterpolated from.
! 
! coordinates = spatial coordinates
       implicit none
       integer, intent(in) :: nonods_l, ndim, nscalar
       integer, intent(in) :: nonods
       real, intent(out) :: x_l(ndim,nonods_l), conc_l(nscalar,nonods_l)
       real, intent(in) :: x(ndim,nonods), conc(nscalar,nonods)
! 
! local variables...
       integer :: nod, nod_l, nod_l_last, nod_prev, nod_next, nod_prev_keep
       real :: weight_interp
       real, allocatable :: x_regular(:), x_l_regular(:)
       integer, allocatable :: nod_prev_list(:)
! 
       allocate(x_regular(nonods), x_l_regular(nonods_l))
       allocate(nod_prev_list(nonods_l))

       do nod = 1,nonods
          x_regular(nod)=real(nod-1)/real(nonods-1)
       end do
       do nod_l = 1,nonods_l
          x_l_regular(nod_l)=real(nod_l-1)/real(nonods_l-1)
       end do

!       print *,'here1'
! for nod_prev...
       nod_prev_keep=1
       nod_prev_list(1)=1
       do nod_l = 2,nonods_l-1
          nod_prev=nod_prev_keep
          do nod = nod_prev,nonods-1
             if(  (x_l_regular(nod_l)>=x_regular(nod)   ) &
             .and.(x_l_regular(nod_l)<=x_regular(nod+1) )   ) then
                nod_prev_list(nod_l)=nod
                nod_prev_keep=nod
                exit
             endif 
          end do
       end do
       nod_prev_list(nonods_l)=nonods-1
! 
!       print *,'here2'
! Form min value
! 
! Calculate x_l from x conc_l from conc by linearly interpolating from regular grid...
       do nod_l = 1,nonods_l
          nod_prev=nod_prev_list(nod_l) 
          nod_next=nod_prev+1
          weight_interp = (x_l_regular(nod_l)  - x_regular(nod_prev)) &
                        / (x_regular(nod_next) - x_regular(nod_prev))
          weight_interp = max( min(weight_interp,1.0), 0.0)
          x_l(:,nod_l)= &
          (1.0-weight_interp) * x(:,nod_prev)    + weight_interp * x(:,nod_next)
          conc_l(:,nod_l)= &
          (1.0-weight_interp) * conc(:,nod_prev) + weight_interp * conc(:,nod_next)
       end do
       x_l(:,nonods_l)=x(:,nonods)
       conc_l(:,nonods_l)=conc(:,nonods)
! 
!       do nod_l = 1,nonods_l
!          print *,'nod_l,nod_prev_list(nod_l),x_l(:,nod_l):', &
!                   nod_l,nod_prev_list(nod_l),x_l(:,nod_l)
!       end do
       return
       end subroutine x_conv_fixed_length
! 
! 

      
