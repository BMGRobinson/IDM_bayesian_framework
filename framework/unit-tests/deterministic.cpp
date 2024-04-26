class DeterministicTest : public :: testing::Test {
protected:
	virtual void SetUp() {
		//State
		double dt = 0.1;
		colvec::fixed<2> u = {2.0,3.0};
		mat::fixed<1,1> R = {0.5};
		ss = statespace(f,h,loglik,dt,2,1);
		ss.setMeasCov(R);
		myse = Deterministic(u,ss);
	}
	Deterministic myse;
	statespace ss;
};

TEST_F(DeterministicTest, forecast) {
	colvec temp;
	myse.forecast(temp);
	colvec mean = myse.getState();
	colvec trueMean = colvec(2);
	trueMean[0] = 2.3;
	trueMean[1] = -7.9;
	ASSERT_TRUE(compareMatrix(mean,trueMean));
}
