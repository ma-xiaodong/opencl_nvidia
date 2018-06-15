build:
	g++ -DLOCAL gemm.cpp -lOpenCL -o gemm_local
	g++ -DLCL_WPT gemm.cpp -lOpenCL -o gemm_lclwpt
	g++ -DLCL_VEC gemm.cpp -lOpenCL -o gemm_lclvec
build_verify:
	g++ -DCMP_RLT -DLOCAL gemm.cpp -lOpenCL -o gemm_local
	g++ -DCMP_RLT -DLCL_WPT gemm.cpp -lOpenCL -o gemm_lclwpt
	g++ -DCMP_RLT -DLCL_VEC gemm.cpp -lOpenCL -o gemm_lclvec
run:
	./gemm_local
	./gemm_lclwpt
	./gemm_lclvec
