
#include <yarp/os/RFModule.h>
#include <yarp/dev/all.h>
#include <cv.h>
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
#include "opencv2/nonfree/nonfree.hpp"
#include <fstream>
using namespace std;
using namespace yarp::os;
using namespace yarp::dev;
using namespace yarp::sig;
using namespace cv;

#define harriscorner2_maxCorners 200
#define harriscorner2_qualityLevel 0.05
#define harriscorner2_minDistance 5.0
#define harriscorner2_blockSize 7
#define harriscorner2_k 0.04
#define Box_size_value 25
#define trainingdatasize_value 3500
#define total_frames_eval_value 100
#define total_no_of_velocityArray 6
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


class Utilities {
      public :
	double findSig ( int j, double fu,double fv, std::vector<cv::Point2f> cornersL1Local,std::vector<cv::Point2f> cornersL2Local)
    {        double u,sigu;
            if (cornersL2Local.size()){
	double i2x , i2y , i1x , i1y ;
	u = 0;
        for ( int i=0; i < cornersL2Local.size(); i++ )
        {
	   i2x = (double)cornersL2Local[i].x;
	   i2y = (double)cornersL2Local[i].y;
	   i1x = (double)cornersL1Local[i].x;
	   i1y = (double)cornersL1Local[i].y;
	    if ( j == 1 )
                u = u + pow ( ( i2x - i1x ) - fu, 2.0 );// * ( ( cornersL2[i].x - cornersL1[i].x ) - mu ) ;
            if ( j == 2 )
                u = u + pow ( ( i2y - i1y ) - fv, 2.0 );// * ( ( cornersL2[i].y - cornersL1[i].y ) - mv ) ;
            if ( j == 3 )
                u = u + ((( i2x - i1x ) - fu) * (( i2y - i1y ) - fv));
            if ( j == 4 )
                u = u + (( i2y - i1y ) - fv) * (( i2x - i1x ) - fu);
        }
        sigu = u/cornersL2Local.size();
	    }
	    else{
	      sigu = 0;
	    }
	return sigu;
    }
    
        double findMu ( int j,std::vector<cv::Point2f> cornersL1Local,std::vector<cv::Point2f> cornersL2Local)
    {        double u,mu;
      if (cornersL2Local.size()){
	double i2x , i2y , i1x , i1y ;
	u = 0;
        for ( int i=0; i < cornersL2Local.size(); i++ )
        {
	   i2x = (double)cornersL2Local[i].x;
	   i2y = (double)cornersL2Local[i].y;
	   i1x = (double)cornersL1Local[i].x;
	   i1y = (double)cornersL1Local[i].y;
            if ( j == 1 )
                u += ( i2x - i1x);
            if ( j == 2 )
                u += ( i2y - i1y );
        }
        mu = u/cornersL2Local.size();
	}
	else {
	  mu = 0;
	}
// 	cout << u << " "<< mu << " " << cornersL2Local.size() << endl << flush;
        return mu;
    }
};


class MyModule:public RFModule
{
    BufferedPort<ImageOf<PixelRgb>  >  imageLIn;
    BufferedPort<ImageOf<PixelRgb>  >  imageLOut;
        BufferedPort<Bottle> inPPort;
    Port port;
    cv::Mat src_gray;
    double bottlevector1[total_no_of_velocityArray];
    double bottlevector2[total_no_of_velocityArray];
    double velocityArray[total_no_of_velocityArray];
    cv::Mat imageL1;
    cv::Mat imageL2;
    int lock;
    double mu,mv,siguu,siguv,sigvv,sigvu;
    ofstream file;


    Utilities t;

    double threshold;
    int hm_check;
    Bottle head_move_check;
    int total_anom_count;
    int total_frames_eval;
    int check_predict_type;
    int first;
    int maxCorners;
        double qualityLevel;
        double minDistance;
        int blockSize ;
        double k ;
    int Box_size, trainingdatasize;
    int velocityArray_pos[total_no_of_velocityArray];
public:
    double getPeriod()
    {
        return 0.0;
    }

    void returnVelocity(int i)
    {
      Bottle *message=inPPort.read();
      for (int n = 0; n < total_no_of_velocityArray; n ++){
	if (i == 1){
      bottlevector1[n] = message->get(n).asDouble();}
      else if (i == 2){bottlevector2[n] = message->get(n).asDouble();}}
      return;
    }
    void imageAcq()
    {
        if (imageL1.data == NULL)
        {
            imageL1 = ( IplImage * ) imageLIn.read()->getIplImage();
	    returnVelocity(1);
            imageL2 = ( IplImage * ) imageLIn.read()->getIplImage();
            returnVelocity(2);
        }
        else
        {
            imageL1 = imageL2;
	    for (int n = 0; n < total_no_of_velocityArray; n ++){
            bottlevector1[n] = bottlevector2[n];}
            imageL2 = ( IplImage * ) imageLIn.read()->getIplImage();
	    returnVelocity(2);
        }
    }

    void findDiffVelocity()
    {
      for (int n = 0; n < total_no_of_velocityArray; n ++){
        if ( bottlevector1[n] != 0 and bottlevector2[n] != 0 )
        {
	  velocityArray[n] = bottlevector2[n] - bottlevector1[n];
        }
        else{
	  velocityArray[n] = 0;
	}
      }
    }

    
    bool updateModule()
    {
      
      
        if ( imageLIn.getInputCount() >0 )
        {
// 	  		  cout<<"acquiring image";
// 		  cout <<endl<<flush;
            imageAcq();
	    Mat tempimageL1, tempimageL2;
	    imageL1.copyTo(tempimageL1);
	    imageL2.copyTo(tempimageL2);
            findDiffVelocity();
// 	    		  cout<<"found image";
// 		  cout <<endl<<flush;
            ImageOf<PixelRgb> &outLImg = imageLOut.prepare();
// 	    		  cout<<"In to harris";
// 		  cout <<endl<<flush;
            // *************** Obtain feature points ************************  Feature extracting
            std::vector<cv::Point2f> cornersL2 = harrisCorner2( imageL2 );
            // ***************************************** Tracking ***************************
            std::vector<uchar> features_found;
            std::vector<float> feature_errors;
            std::vector<cv::Point2f> cornersL1;
	    cv::Mat cornersL1dummy;
            cv::Mat imageL11;
            cv::Mat imageL22;
            cv::cvtColor ( imageL2, imageL22, CV_BGR2GRAY );
            cv::cvtColor ( tempimageL1, imageL11, CV_BGR2GRAY );
// 	    		  cout<<"into tracker";
// 		  cout <<endl<<flush;
		  
// 	            calcOpticalFlowPyrLK ( imageL22, imageL11, cornersL2dummy, cornersL1dummy, features_found, feature_errors ,Size ( 101, 101 ), 7,cvTermCriteria ( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );
		  
	    calcOpticalFlowFarneback(imageL22, imageL11, cornersL1dummy, 0.5, 5, 51, 3, 5, 1.2, 0);

// 	    		  cout<<"capture the track";
// 		  cout <<endl<<flush;
		  
// 	for ( int i=0; i < features_found.size(); i++ )
//             {
// 	      if(features_found[i] != 0 && feature_errors[i] < 550){
// 	      cornersL1.push_back(cornersL1dummy[i]);
// 	      cornersL2.push_back(cornersL2dummy[i]);
// 	      }
//             }
// 	            for ( int i=0; i < cornersL2.size(); i++ )
//             {
//                 circle ( tempimageL1, Point ( cornersL2[i].x, cornersL2[i].y ), 5, CV_RGB ( 255,0,0 ), 2, 8, 0 );
//             }
            
            
//             for ( int i=0; i < cornersL2.size(); i++ )
//             {
//                 Point p0 ( ceil ( cornersL2[i].x ), ceil ( cornersL2[i].y ) );
//                 Point p1 ( ceil ( cornersL1[i].x ), ceil ( cornersL1[i].y ) );
//                 line ( tempimageL1 , p0, p1, CV_RGB ( 0,255,255 ), 2 );
//             }
	   cornersL1 = drawOptFlowMapsparse(cornersL1dummy,cornersL2, tempimageL1);
            IplImage tmp = tempimageL1;
            outLImg.resize ( tmp.width, tmp.height );
            cvCopyImage ( &tmp, ( IplImage * ) outLImg.getIplImage() );
            imageLOut.write();
	      if (lock == 0){
	     cout<<"Collecting Data ...";
	     cout <<endl<<flush;}
	      mu = 0;
	      mv = 0;
	      siguu = 0;
	      sigvv = 0;
	      siguv = 0;
// 	    for ( int i=0; i < cornersL1.size(); i++ )
//             {
// 	      cout << cornersL1[i]<<endl<<flush;
// 	      cout << cornersL2[i]<<endl<<flush;
//             }

                if ( lock < trainingdatasize )
                {
                    mu = t.findMu ( 1,cornersL1,cornersL2 );
                    mv = t.findMu ( 2,cornersL1,cornersL2 );
                    siguu = t.findSig ( 1,mu,mv,cornersL1,cornersL2 );
                    sigvv = t.findSig ( 2,mu,mv,cornersL1,cornersL2 );
                    siguv = t.findSig ( 3,mu,mv,cornersL1,cornersL2 );
		    if(velocityArray[0] != 0){
                    writeDataToFile ();
                    lock ++;
		    }
                }
                else
		{
		  if (!first){
		  cout<<"Data Collected for Training";
		  cout <<endl<<flush;
		  first = 1;
		  }
		}
        }
        return true;
    }

    std::vector<cv::Point2f> harrisCorner2 ( cv::Mat src )
    {
        cv::Mat src2;
        cv::cvtColor ( src, src2, CV_BGR2GRAY );
        std::vector<cv::Point2f> cornersA;
        goodFeaturesToTrack( src2,cornersA,maxCorners,qualityLevel,minDistance,cv::Mat(),blockSize,false,k);
        return cornersA;
    }

  std::vector<cv::Point2f> drawOptFlowMapsparse (const cv::Mat& flow,const std::vector<cv::Point2f>& harris, cv::Mat& cflowmap) {
	      std::vector<cv::Point2f> cornersA;
	     cv::Point2f cornerAPoint;
           for ( int i=0; i < harris.size(); i++ )
            {
                circle ( cflowmap, Point ( harris[i].x, harris[i].y ), 5, CV_RGB ( 255,0,0 ), 2, 8, 0 );
            }
	      
	      
            for ( int i=0; i < harris.size(); i++ )
            {
	      const Point2f& fxy = flow.at<Point2f>(harris[i].y, harris[i].x);
                Point p0 ( ceil ( harris[i].x ), ceil ( harris[i].y ) );
                Point p1 ( ceil ( harris[i].x + fxy.x), ceil ( harris[i].y + fxy.y) );
                line ( cflowmap , p0, p1, CV_RGB ( 0,255,255 ), 2 );
		cornerAPoint.x = harris[i].x + fxy.x;
		cornerAPoint.y = harris[i].y + fxy.y;
		cornersA.push_back(p1);
            }
        return cornersA;
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
        string moduleName = "collect";
        string imageIn = "/" + moduleName + "/image:i";
        imageLIn.open ( imageIn.c_str() );
        string imageOut = "/" + moduleName + "/image:o";
        imageLOut.open ( imageOut.c_str() );
	inPPort.open(("/"+moduleName+"/pos:i").c_str());
        lock = 0;
	file.open("velocityArray.csv", std::fstream::trunc);
	file.close();
	
	file.open("dataFile.csv", std::fstream::trunc);
	file.close();
	file.open("flowstatistics_train_mu.csv", std::fstream::trunc);
	file.close();
	file.open("flowstatistics_train_mv.csv", std::fstream::trunc);
	file.close();
	file.open("flowstatistics_train_siguu.csv", std::fstream::trunc);
	file.close();
	file.open("flowstatistics_train_sigvv.csv", std::fstream::trunc);
	file.close();
	file.open("flowstatistics_train_siguv.csv", std::fstream::trunc);
	file.close();
	first = 0;
	total_frames_eval = total_frames_eval_value;
         maxCorners = harriscorner2_maxCorners;
         qualityLevel = harriscorner2_qualityLevel;
         minDistance = harriscorner2_minDistance;
         blockSize = harriscorner2_blockSize;
         k = harriscorner2_k;
	 trainingdatasize = trainingdatasize_value;
	 total_frames_eval = total_frames_eval_value;
	 velocityArray_pos[0] = 0;
	 velocityArray_pos[1] = 1;
	 velocityArray_pos[2] = 2;
	 velocityArray_pos[3] = 3;
	 velocityArray_pos[4] = 4;
	 velocityArray_pos[5] = 5;
	 
	return true;
  }
  
  
    void writeDataToFile ( )
    {
// 	if (mu && mv && siguu && sigvv && siguv){
// 	for (int n = 0; n < total_no_of_velocityArray; n ++){
// 	    if (!velocityArray[n]){
// 	     return;
// 	    }
// 	}
	
// 	cout<<mu<<" "<<mv<<" "<<siguu<<" "<<sigvv<<" "<<siguv<<endl<<flush;
// 	cout<<"train"<<addtest<<endl<<flush;
	file.open("velocityArray.csv", std::fstream::app);
// 	cout << "total_no_of_velocityArray : " << total_no_of_velocityArray << endl << flush; 
	for (int n = 0; n < total_no_of_velocityArray; n ++){
// 	  cout<<velocityArray[n]<<" ";
        file<<velocityArray[n]<<" ";
	}
// 	file<<mu<<",";
// 	file<<mv<<",";
// 	file<<siguu<<",";
// 	file<<sigvv<<",";
// 	file<<siguv;
// 	cout<<endl;
	file<<endl;
	file.close();
      	file.open("flowstatistics_train_mu.csv", std::fstream::app);
        file<<mu<<endl;
	file.close();
	file.open("flowstatistics_train_mv.csv", std::fstream::app);
        file<<mv<<endl;
	file.close();
	file.open("flowstatistics_train_siguu.csv", std::fstream::app);
        file<<siguu<<endl;
	file.close();
	file.open("flowstatistics_train_sigvv.csv", std::fstream::app);
        file<<sigvv<<endl;
	file.close();
	file.open("flowstatistics_train_siguv.csv", std::fstream::app);
        file<<siguv<<endl;
	file.close();
	file.open("dataFile.csv", std::fstream::app);
      for (int n = 0; n < total_no_of_velocityArray; n ++){
// 	  cout<<velocityArray[n]<<" ";
        file<<velocityArray[n]<<",";
	}
	file<<mu<<",";
	file<<mv<<",";
	file<<siguu<<",";
	file<<sigvv<<",";
	file<<siguv;
	file<<endl;
	file.close();
// 	cout<<endl;
// 	}
    }


    bool interruptModule()
    {
        imageLIn.interrupt();
        imageLOut.interrupt();
	inPPort.interrupt();
        return true;
    }

    bool close()
    {
        imageLIn.close();
        imageLOut.close();
	inPPort.close();
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

