
#include <yarp/os/RFModule.h>
#include "gurls++/gurls.h"
#include <yarp/dev/all.h>
#include <ml.h>
#include <yarp/dev/Drivers.h>
#include <yarp/dev/PolyDriver.h>

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <iostream>
#include <math.h>
#include <yarp/os/Network.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/Time.h>
#include <yarp/sig/Vector.h>
#include <yarp/os/Semaphore.h>

using namespace std;
using namespace yarp::os;
using namespace yarp::dev;
using namespace yarp::sig;
using namespace gurls;

typedef double T;


class ML1{
//       GURLS G;
      GurlsOptionsList* opt;
      OptTaskSequence *seq;
      GurlsOptionsList * process, *paramsel;
      OptString* hofun;
      OptFunction* optfun;
public:
  
  
  
    yarp::sig::Vector trainData1(string flowstatistics_train_name){
    int blockSize = 10;
    cout<<"Inside Training"<<endl<<flush;
   yarp::sig::Vector error_graph;


    opt = new GurlsOptionsList("IndependentMotionDetection", true);
    seq = new OptTaskSequence();

    
    *seq << "paramsel:siglam" << "kernel:rbf"<<"optimizer:rlsdual";
    *seq << "predkernel:traintest" << "pred:dual" << "perf:rmse";
    opt->addOpt("seq", seq);

    process = new GurlsOptionsList("processes", false);
    OptProcess* process1 = new OptProcess();
    

    *process1 << GURLS::computeNsave  << GURLS::computeNsave << GURLS::computeNsave;
    *process1 << GURLS::ignore << GURLS::ignore<< GURLS::ignore;
    process->addOpt("one", process1);

    OptProcess* process2 = new OptProcess();
    
    
    *process2 << GURLS::load  << GURLS::load << GURLS::load;
    *process2 << GURLS:: computeNsave << GURLS:: computeNsave << GURLS:: computeNsave;
    process->addOpt("two", process2);

    opt->addOpt("processes", process);
    hofun = new OptString("rmse");

    opt->removeOpt("hoperf");
    opt->addOpt("hoperf", hofun);

    
    OptNumber* nlambda = new OptNumber (2);
    opt->removeOpt("nlambda");
    opt->addOpt("nlambda", nlambda);
    
    
    OptNumber* nsigma = new OptNumber (2);
    opt->removeOpt("nsigma");
    opt->addOpt("nsigma", nsigma);

    
    gMat2D<T> ML_velocityArray,ML_velocityArray_test, ML_flowstatistics_train, ML_flowstatistics_test;
    ML_velocityArray.readCSV("velocityArray.csv");
    ML_flowstatistics_train.readCSV(flowstatistics_train_name+".csv");
    GURLS G;
    string jobId0("one");
    G.run(ML_velocityArray, ML_flowstatistics_train, *opt, jobId0); 
    opt->printAll();
    return error_graph;
  }

  
  yarp::sig::Vector trainData(string flowstatistics_train_name){
    int blockSize = 10;
    cout<<"Inside Training"<<endl<<flush;
   yarp::sig::Vector error_graph;
    opt = new GurlsOptionsList(flowstatistics_train_name, true);
    seq = new OptTaskSequence();
    *seq << "paramsel:siglam" << "kernel:rbf"<<"optimizer:rlsdual";
    *seq << "predkernel:traintest" << "pred:dual" << "perf:rmse";
    opt->addOpt("seq", seq);
    process = new GurlsOptionsList("processes", false);
    OptProcess* process1 = new OptProcess();
    *process1 << GURLS::computeNsave  << GURLS::computeNsave << GURLS::computeNsave;
    *process1 << GURLS::ignore << GURLS::ignore<< GURLS::ignore;
    process->addOpt("one", process1);
    OptProcess* process2 = new OptProcess();
    *process2 << GURLS::load  << GURLS::load << GURLS::load;
    *process2 << GURLS:: computeNsave << GURLS:: computeNsave << GURLS:: computeNsave;
    process->addOpt("two", process2);
    opt->addOpt("processes", process);
    hofun = new OptString("rmse");
    opt->removeOpt("hoperf");
    opt->addOpt("hoperf", hofun);
          gMat2D<T> ML_velocityArray,ML_velocityArray_test, ML_flowstatistics_train, ML_flowstatistics_test;
    ML_velocityArray.readCSV("velocityArray.csv");
    ML_flowstatistics_train.readCSV(flowstatistics_train_name+".csv");
    GURLS G;
    string jobId0("one");
    G.run(ML_velocityArray, ML_flowstatistics_train, *opt, jobId0);
	opt->printAll();
	return error_graph;
  }

  gMat2D<T> predictML(gMat2D<T> ML_velocityArray_test){
//     gMat2D<T> ML_velocityArray_test(1,sizeof(velocityArray)/8);
    gMat2D<T> ML_flowstatistics_test(1,1);
//     ML_velocityArray_test=velocityArray;
    ML_flowstatistics_test=0;
    string jobId1("two");
      GURLS G;
    G.run(ML_velocityArray_test, ML_flowstatistics_test, *opt, jobId1);
    const gMat2D<T>& pred_mat = OptMatrix<gMat2D<T> >::dynacast(opt->getOpt("pred"))->getValue();
    return pred_mat;
  }
};



class MyModule:public RFModule
{
       ML1 mlu, mlv, mlsiguu, mlsigvv ,mlsiguv;
public:
    double getPeriod()
    {
        return 0.0;
    }

    bool updateModule()
    {

        mlu.trainData("flowstatistics_train_mu");

	mlv.trainData("flowstatistics_train_mv");

	mlsiguu.trainData("flowstatistics_train_siguu");

	mlsigvv.trainData("flowstatistics_train_sigvv");

       mlsiguv.trainData("flowstatistics_train_siguv");
        return true;
    }

    


    bool respond ( const Bottle& command, Bottle& reply )
    {
        cout<<"Got something, echo is on"<<endl;
        if ( command.get ( 0 ).asString() =="quit" )
            return false;
        else
            reply=command;
        return true;
    }

   virtual bool configure(ResourceFinder &rf)
    {
	return true;
  }
  

    bool interruptModule()
    {
        return true;
    }

    bool close()
    {
        return true;
    }
};

int main ( int argc, char *argv[] )
{
    Network yarp;
    if (!yarp.checkNetwork())
        return -1;
    MyModule mod;
//     YARP_REGISTER_DEVICES(icubmod)
    ResourceFinder rf;
    rf.configure ( argc, argv );
    rf.setVerbose ( true );
    return mod.runModule(rf);
    return 0;
}

