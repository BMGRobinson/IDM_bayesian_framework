#include <gtest/gtest.h>
#include <armadillo>
#include <mpi.h>
#include <cmath>
using namespace arma;

/* For libconfig */
#include <libconfig.h++>
using namespace libconfig;



/*****************************************************************
******************************************************************
* Following is code to make the output readable when running MPI. Adapted from gtest.cc
******************************************************************
*****************************************************************/

using ::testing::EmptyTestEventListener;
using ::testing::TestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestCase;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;
using ::testing::FLAGS_gtest_color;
using ::testing::FLAGS_gtest_filter;
using ::testing::FLAGS_gtest_repeat;
using ::testing::internal::String;
//using namespace ::testing::internal;

//My own listener that attach itself to the one google made so that only root prints it
class MPI_RESULT_PRINTER : public testing::TestEventListener {
public:
	explicit MPI_RESULT_PRINTER(TestEventListener* PrettyResultPrinter): PrettyResultPrinter(PrettyResultPrinter) {}
	void OnTestProgramStart(const testing::UnitTest& unit_test) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnTestProgramStart(unit_test);
		}
	}
	void OnTestIterationStart(const testing::UnitTest& unit_test, int iteration) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnTestIterationStart(unit_test, iteration);
		}
	}
	void OnEnvironmentsSetUpStart(const testing::UnitTest& unit_test) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnEnvironmentsSetUpStart(unit_test);
		}
	}
	void OnEnvironmentsSetUpEnd(const testing::UnitTest& unit_test) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnEnvironmentsSetUpEnd(unit_test);
		}
	}
	void OnTestCaseStart(const testing::TestCase& test_case) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnTestCaseStart(test_case);
		}
	}
	void OnTestStart(const testing::TestInfo& test_info) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnTestStart(test_info);
		}
	}
	void OnTestPartResult(const testing::TestPartResult& test_part_result) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnTestPartResult(test_part_result);
		}

	}
	void OnTestEnd(const testing::TestInfo& test_info) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnTestEnd(test_info);
		}
	}
	void OnTestCaseEnd(const testing::TestCase& test_case) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnTestCaseEnd(test_case);
		}
	}
	void OnEnvironmentsTearDownStart(const testing::UnitTest& unit_test) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnEnvironmentsTearDownStart(unit_test);
		}
	}
	void OnEnvironmentsTearDownEnd(const testing::UnitTest& unit_test) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnEnvironmentsTearDownEnd(unit_test);
		}
	}
	void OnTestIterationEnd(const testing::UnitTest& unit_test, int iteration) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnTestIterationEnd(unit_test, iteration);
		}
	}
	void OnTestProgramEnd(const testing::UnitTest& unit_test) {
		if ( MPI::COMM_WORLD.Get_rank() == 0 ) {
			PrettyResultPrinter->OnTestProgramEnd(unit_test);
		}
	}
private:
	// gtest PrettyUnitTestResultPrinter
	std::unique_ptr<TestEventListener> PrettyResultPrinter;
	MPI_RESULT_PRINTER(const MPI_RESULT_PRINTER&) = delete;
	MPI_RESULT_PRINTER& operator=(const MPI_RESULT_PRINTER&) = delete;
};


int main(int argc, char **argv) {
	::testing::InitGoogleTest( &argc, argv );
	UnitTest& unit_test = *UnitTest::GetInstance();
	TestEventListeners& listeners = unit_test.listeners();
	//delete listeners.Release(listeners.default_result_printer());
	listeners.Append(new MPI_RESULT_PRINTER( listeners.Release(listeners.default_result_printer()) ) );

	/* Set the seed to a random value, important otherwise each process have the same seed (and same samples)*/
	arma_rng::set_seed_random();

	MPI::Init();
	bool result = RUN_ALL_TESTS();
	MPI::COMM_WORLD.Bcast(&result, 1, MPI::BOOL, 0);
	MPI::Finalize();
	return 0;
}
