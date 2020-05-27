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

public class G48HW3 {

    static long SEED=1231829;
    static Random generator=new Random(SEED);


    //Method which given a path to a CSV file returns an ArrayList list of spark org.apache.spark.mllib package.Spark
    private static ArrayList<Vector> readCSV(String path) throws FileNotFoundException {
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


    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            throw new IllegalArgumentException("USAGE: only 1 parameter required.");
        }
        SparkConf conf = new SparkConf(true).setAppName("Homework3").setMaster("local").set("spark.testing.memory", "2147480000");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");
        // Read number of partitions

        String path = args[0];
        ArrayList<Vector> a = runSequential(readCSV(path), 10);

        System.out.println(a);

    }




    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

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