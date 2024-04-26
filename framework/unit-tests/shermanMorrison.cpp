class SMTest : public :: testing::Test {
protected:
	virtual void SetUp() {
		//1x1 Tests
		A1x1 = zeros<mat>(1, 1);
		U1x1 = zeros<mat>(1, 1);
		V1x1 = zeros<mat>(1, 1);
		Y1x1 = zeros<mat>(1, 1);
		dpVY1x1 = zeros<vec>(1);

		//1 x 10 Tests
		U1x10 = zeros<mat>(1, 10);
		V1x10 = zeros<mat>(1, 10);
		Y1x10 = zeros<mat>(1, 10);
		dpVY1x10 = zeros<vec>(10);

		//2x2 Tests
		A2x2 = zeros<mat>(2, 2);
		U2x1 = zeros<mat>(2, 1);
		V2x1 = zeros<mat>(2, 1);
		Y2x1 = zeros<mat>(2, 1);

		//2x10 Tests
		U2x10 = zeros<mat>(2, 10);
		V2x10 = zeros<mat>(2, 10);
		Y2x10 = zeros<mat>(2, 10);

		//Right hand side
		b1 = zeros<colvec>(1);
		b2 = zeros<colvec>(2);

		abs_error = 1e-8;

	}
	mat A1x1, A2x2;
	mat U1x1, U1x10, U2x1, U2x10;
	mat V1x1,V1x10,V2x1, V2x10;
	mat Y1x1, Y1x10, Y2x1, Y2x10;
	vec dpVY1x10, dpVY1x1;
	colvec b1, b2;
	double abs_error;
};

TEST_F(SMTest, SolveingAxb_scalar) {
	A1x1(0,0) = 2.0;
	b1(0) = 1.0;
	colvec actual = zeros<colvec>(1);
	//Setting up Sherman-Morrison
	shermanMorrisonMPIPreProc(A1x1, U1x1, V1x1, Y1x1, dpVY1x1, MPI::COMM_WORLD);
	simplifiedShermanMorrison(A1x1, V1x1, Y1x1, dpVY1x1, actual, b1, MPI::COMM_WORLD);
	ASSERT_NEAR(actual.at(0), 0.5, abs_error);
}

TEST_F(SMTest, SolveingApUVxb_scalar) {
	A1x1(0,0) = 2.0;
	b1(0) = 1.0;
	U1x1(0,0) = 1.0;
	V1x1(0,0) = 2.0;
	colvec actual = zeros<colvec>(1);
	//Setting up Sherman-Morrison
	shermanMorrisonMPIPreProc(A1x1, U1x1, V1x1, Y1x1, dpVY1x1, MPI::COMM_WORLD);
	simplifiedShermanMorrison(A1x1, V1x1, Y1x1, dpVY1x1, actual, b1, MPI::COMM_WORLD);
	ASSERT_NEAR(actual.at(0), 0.25, abs_error);
}

TEST_F(SMTest, SolveingApUV10xb_scalar) {
	A1x1(0,0) = 2.0;
	b1(0) = 1.0;
	for (int i = 0; i < 10; i ++) {
		U1x10(0,i) = double(i);
		V1x10(0,i) = 2.0*double(i);
	}
	colvec actual = zeros<colvec>(1);
	//Setting up Sherman-Morrison
	shermanMorrisonMPIPreProc(A1x1, U1x10, V1x10, Y1x10, dpVY1x10, MPI::COMM_WORLD);
	simplifiedShermanMorrison(A1x1, V1x10, Y1x10, dpVY1x10, actual, b1, MPI::COMM_WORLD);
	ASSERT_NEAR(actual.at(0), 1.0 / 572.0 , abs_error);
}

TEST_F(SMTest, SolveingAxb_vector) {
	A2x2(0,0) = 2.0;
	A2x2(1,1) = 0.5;
	b2(0) = 1.0;
	b2(1) = 2.0;
	colvec actual = zeros<colvec>(2);
	//Setting up Sherman-Morrison
	shermanMorrisonMPIPreProc(A2x2, U2x1, V2x1, Y2x1, dpVY1x1, MPI::COMM_WORLD);
	simplifiedShermanMorrison(A2x2, V2x1, Y2x1, dpVY1x1, actual, b2, MPI::COMM_WORLD);
	ASSERT_NEAR(actual.at(0), 0.5, abs_error);
	ASSERT_NEAR(actual.at(1), 4.0, abs_error);
}

TEST_F(SMTest, SolveingApUV10xb_vector) {
	A2x2(0,0) = 2.0;
	A2x2(1,1) = 0.5;


	for (int i = 0; i < 10; i ++) {
		U2x10(0,i) = double(i);
		U2x10(1,i) = double(i)*0.5+1.0;
		V2x10(0,i) = 2.0*double(i);
		V2x10(1,i) = 1.0 - 2.0*double(i);
	}

	b2(0) = 1.0;
	b2(1) = 2.0;
	colvec actual = zeros<colvec>(2);
	//Setting up Sherman-Morrison
	shermanMorrisonMPIPreProc(A2x2, U2x10, V2x10, Y2x10, dpVY1x10, MPI::COMM_WORLD);
	simplifiedShermanMorrison(A2x2, V2x10, Y2x10, dpVY1x10, actual, b2, MPI::COMM_WORLD);
	ASSERT_NEAR(actual.at(0), 0.56594724, abs_error);
	ASSERT_NEAR(actual.at(1), 0.61470823, abs_error);
}

TEST_F(SMTest, SolveingApUVxb_vector) {
	A2x2(0,0) = 2.0;
	A2x2(1,1) = 0.5;
	U2x1(0,0) = 2.0;
	U2x1(1,0) = 1.0;

	V2x1(0,0) = 1.0;
	V2x1(1,0) = 1.0;

	b2(0) = 1.0;
	b2(1) = 2.0;
	colvec actual = zeros<colvec>(2);
	//Setting up Sherman-Morrison
	shermanMorrisonMPIPreProc(A2x2, U2x1, V2x1, Y2x1, dpVY1x1, MPI::COMM_WORLD);
	simplifiedShermanMorrison(A2x2, V2x1, Y2x1, dpVY1x1, actual, b2, MPI::COMM_WORLD);
	ASSERT_NEAR(actual.at(0), -5.0/8.0, abs_error);
	ASSERT_NEAR(actual.at(1), 14.0/8.0, abs_error);
}
