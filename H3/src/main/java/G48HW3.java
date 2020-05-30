//%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&%%%%%%%%%%%%%
//%%%%%%% Comments about the results %&%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//The Exact algorithm was clearly the slowest one but as the name says it returns the best
//possible solution.
//The 2-approximation algorithm is strictly dependent on the value of k given as input. By increasing the values of k
//we get higher execution times but at the same time better approximations.
//The k-center-based algorithm seems to be best compromise, it gets very good approximations with low execution times
//even with small values of k

import com.google.inject.internal.asm.$ByteVector;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

import static java.lang.Math.sqrt;

public class G48HW3 {

    static long SEED=1231829;
    static Random generator=new Random(SEED);

    static final SparkConf conf = new SparkConf(true).setAppName("Homework3").setMaster("local").set("spark.testing.memory", "2147480000");
    static final  JavaSparkContext sc = new JavaSparkContext(conf);

    //implementation of the kcenter algorithm
    private static Iterator<Vector> farthestFirstTraversal(Iterator<Vector> pointsRDD, int k){
        //Data structures initialization
        ArrayList<Vector> s_copy= new ArrayList<>();
        pointsRDD.forEachRemaining(s_copy::add);
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

        return centers.iterator();
    }


    public static Vector readPoint(String point) {
        return Vectors.dense(Arrays.stream(point.split(",")).mapToDouble(Double::parseDouble).toArray());
    }


    public static double measure(List<Vector> pointSet) {
        double sum = 0;
        for(Vector point1: pointSet)
            for(Vector point2: pointSet)
                sum += Vectors.sqdist(point1, point2);
        double k = pointSet.size();
        return sum / ((k * ( k - 1 )) / 2);
    }

    public static void runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L) {
        // Calculate time
        long startTime = System.currentTimeMillis();
        // Repartition the JavaRDD
        JavaRDD<Vector> points = pointsRDD.repartition(L).cache();
        System.out.println("Number of points = " + pointsRDD.collect().size() + "\nk = " + k + "\nL = " + L + "\nInitialization time = " + (System.currentTimeMillis() - startTime) + " ms" );
        startTime = System.currentTimeMillis();
        List<Vector> coreset = points.mapPartitions(partition -> farthestFirstTraversal(partition, k)).collect();
        System.out.println("Runtime of Round 1 = " + (System.currentTimeMillis() - startTime) + " ms");
        startTime = System.currentTimeMillis();
        List<Vector> centers = runSequential(coreset, k);
        System.out.println("Runtime of Round 2 = " + (System.currentTimeMillis() - startTime) + " ms\nAverage distance = " + measure(centers));
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: needed path, k, L.");
        }
        sc.setLogLevel("WARN");
        // Read inputs
        String path = args[0];
        int k = Integer.parseInt(args[1]);
        int L = Integer.parseInt(args[2]);
        JavaRDD<Vector> pointsRDD = sc.textFile(path).map(G48HW3::readPoint);
        runMapReduce(pointsRDD, k, L);

        //ArrayList<Vector> a = runSequential(readCSV(path), 10);

        //System.out.println(a);

    }




    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static List<Vector> runSequential(final List<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    } // END runSequential





}
