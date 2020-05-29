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
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

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

    private static JavaRDD<Vector> farthestFirstTraversal(Iterator<Vector> pointsRDD, int k){
        List<Vector> points = pointsRDD.collect();
        List<Vector> S = new ArrayList<>();
        S.add(points.remove(generator.nextInt(points.size() - 1)));
        for(int i=0; i < k-2; i++) {
            int max = -1;
            double dist = -1;
            for(int c = 0; c < S.size(); c++) {
                for (int p = 0; p < points.size(); p++) {
                    double actDist = Vectors.sqdist(S.get(c), points.get(p));
                    if ( actDist > dist ) {
                        dist = actDist;
                        max = p;
                    }
                }
            }
            S.add(points.remove(max));
        }
        return pointsRDD;
    }

    public static Vector readPoint(String point) {
        return Vectors.dense(Arrays.stream(point.split(",")).mapToDouble(Double::parseDouble).toArray());
    }


    public static void runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L) {
        JavaRDD<String> res = pointsRDD.mapPartitions(partition -> {
            System.out.println("aa");
            farthestFirstTraversal(partition, k);
            return "aa";
        });
        System.out.println("rr");
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: needed path, k, L.");
        }
        SparkConf conf = new SparkConf(true).setAppName("Homework3").setMaster("local").set("spark.testing.memory", "2147480000");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");
        // Read number of partitions

        long startTime = System.currentTimeMillis();
        String path = args[0];
        int k = Integer.parseInt(args[1]);
        int L = Integer.parseInt(args[2]);
        JavaRDD<Vector> pointsRDD = sc.textFile(path).map(t -> readPoint(t)).repartition(L).cache();
        long endTime = System.currentTimeMillis();
        System.out.println("Number of points = " + pointsRDD.collect().size() + "\nk = " + k + "\nL = " + L + "\nInitialization time = " + (endTime - startTime) + " ms" );
        runMapReduce(pointsRDD, k, L);

        //ArrayList<Vector> a = runSequential(readCSV(path), 10);

        //System.out.println(a);

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
