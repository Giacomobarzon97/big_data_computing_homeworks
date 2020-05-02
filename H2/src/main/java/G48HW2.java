//%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&%%%%%%%%%%%%%
//%%%%%%% Comments about the results %&%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//The Exact algorithm was clearly the slowest one but as the name says it returns the best
//possible solution.
//The 2-approximation algorithm is strictly dependent on the value of k given as input. By increasing the values of k
//we get higher execution times but at the same time better approximations.
//The k-center-based algorithm seems to be best compromise, it gets very good approximations with low execution times
//even with small values of k

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import static java.lang.Math.sqrt;

public class G48HW2 {
    //%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%% IMPORTANT %%%%%%%
    //%%%%%%%%%%%%%%%%%%%%%%%%%%
    //Defining the random generator
    //Set the SEED constant as you like
    static long SEED=1231829;
    static Random generator=new Random(SEED);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%% Defining methods %%%%%%%
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    //Method which given a path to a CSV file returns an ArrayList list of spark org.apache.spark.mllib package.Spark
    private static ArrayList<Vector>readCSV(String path) throws FileNotFoundException {
        ArrayList<Vector> records=new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(path))) {
            while (scanner.hasNext()) {
                double[]  values = Arrays.stream(scanner.next().split(","))
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                records.add(Vectors.dense(values));
            }
        }
        return records;
    }
    //Algorithm for calculating the exact maximum distance between two points in the dataset
    private static double exactMPD(ArrayList<Vector> s) throws IOException {
        double max_distance=-1;
        for(int i=0; i<s.size();i++){
            for(int j=i+1; j<s.size();j++){
                double current_distance=sqrt(Vectors.sqdist(s.get(i),s.get(j)));
                if (current_distance>max_distance) {
                    max_distance=current_distance;
                }
            }
        }
        return max_distance;
    }

    //Implementation of the 2-approximation algorithm
    private static double twoApproxMPD(ArrayList<Vector>S, int k){
        //Selecting k random points from the dataset with no repetition
        ArrayList<Vector> s_copy= (ArrayList<Vector>) S.clone();
        ArrayList<Vector> s_subset= new ArrayList<>();
        for (int i=0;i<k;i++){
            int index=generator.nextInt(s_copy.size());
            s_subset.add(s_copy.remove(index));
        }
        //Calculating the max distance between each point of the complete set and the subset extracted above.
        double max_distance=-1;
        for(int i=0; i<S.size();i++){
            for(int j=0; j<s_subset.size();j++){
                double current_distance=sqrt(Vectors.sqdist(S.get(i),s_subset.get(j)));
                if (current_distance>max_distance) {
                    max_distance=current_distance;
                }
            }
        }
        return max_distance;
    }

    //implementation of the kcenter algorithm
    private static ArrayList<Vector> kCenterMPD(ArrayList<Vector>S, int k){
        //Data structures initialization
        ArrayList<Vector> s_copy= (ArrayList<Vector>) S.clone();
        ArrayList<Vector> centers= new ArrayList<>();
        //Extracting a random point from the dataset as the initial center
        int index=generator.nextInt(s_copy.size());
        centers.add(s_copy.remove(index));
        //initializing a data structure for keeping the minimum distance between
        //a point not selected as center and the center subset
        ArrayList<Double> centers_distances=new ArrayList<>();
        for(int i=0; i<s_copy.size();i++){//O(|S|)
            centers_distances.add(sqrt(Vectors.sqdist(s_copy.get(i),centers.get(0))));
        }
        //For cicle which extracts at each iteration the point with the maximum distance
        //to the closest point of the center subset
        //Total Cost: O(k*|S|)
        for (int i=1;i<k;i++){//O(k)
            double best_distance=-1;
            int best_point=-1;
            //Calculating the point which minimizes the distance with the center subset
            for(int j=0;j<s_copy.size();j++){//O(|S|)
                if (centers_distances.get(j)>best_distance){
                    best_distance=centers_distances.get(j);
                    best_point=j;
                }
            }
            Vector new_center=s_copy.remove(best_point);
            centers.add(new_center);
            centers_distances.remove(best_point);
            //Updating the centers_distances data structure
            for(int j=0;j<centers_distances.size();j++){//O(|S|)
                double new_center_distance=sqrt(Vectors.sqdist(s_copy.get(j),new_center));
                if(new_center_distance<centers_distances.get(j)){
                    centers_distances.set(j,new_center_distance);
                }
            }
        }
        return centers;
    }
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%% Main Class %%%%%%%%%%%%%
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");
        // Read number of partitions
        int k=Integer.parseInt(args[0]);
        String path = args[1];
        ArrayList<Vector> inputPoints = readCSV(path);
        double max_distance;
        Instant starting_time,ending_time;

        starting_time=Instant.now();
        max_distance=exactMPD(inputPoints);
        ending_time=Instant.now();
        System.out.println("EXACT ALGORITHM");
        System.out.println("Max distance = "+max_distance);
        System.out.println("Running time = "+ Duration.between(starting_time, ending_time).toMillis()+" ms");
        System.out.println();

        starting_time=Instant.now();
        max_distance=twoApproxMPD(inputPoints,k);
        ending_time=Instant.now();
        System.out.println("2-APPROXIMATION ALGORITHM");
        System.out.println("k = "+k);
        System.out.println("Max distance = "+max_distance);
        System.out.println("Running time = "+ Duration.between(starting_time, ending_time).toMillis()+" ms");
        System.out.println();

        starting_time=Instant.now();
        ArrayList<Vector> centers=kCenterMPD(inputPoints,k);
        max_distance=exactMPD(centers);
        ending_time=Instant.now();
        System.out.println("k-CENTER-BASED ALGORITHM");
        System.out.println("k = "+k);
        System.out.println("Max distance = "+max_distance);
        System.out.println("Running time = "+ Duration.between(starting_time, ending_time).toMillis()+" ms");
    }
}
