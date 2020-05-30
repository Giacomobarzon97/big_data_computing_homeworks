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
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class G48HW3 {

    static long SEED=1231829;
    static Random generator=new Random(SEED);

    static final SparkConf conf = new SparkConf(true).setAppName("Homework3").setMaster("local").set("spark.testing.memory", "2147480000");
    static final  JavaSparkContext sc = new JavaSparkContext(conf);

    private static JavaPairRDD<Vector, ArrayList<Vector>> farthestFirstTraversal(Iterator<Object> pointsRDD, int k){
        List<Vector> S = new ArrayList<>();
        List<Vector> points = new ArrayList<>();
        pointsRDD.forEachRemaining(el -> {points.add((Vector)el);});
        S.add( points.remove(generator.nextInt(points.size() - 1)));

        Map<Vector, Tuple2<Vector, Double>> partition = new HashMap<>();

        // Iterate k-1 times to find other centers
        for(int i=0; i < k-1; i++) {
            int max = -1;
            double dist = -1;
            // Find point pi in points that maximizes d(pi, S)
            // Iterate all centers
            for(int c = 0; c < S.size(); c++) {
                // Iterate all points in P - S
                Vector actCenter = S.get(c);
                for (int p = 0; p < points.size(); p++) {
                    Vector actPoint = points.get(p);
                    double actDist = Vectors.sqdist(actCenter, actPoint);
                    /*
                    * Since in each iteration calculates the distance between each point and each center, keep trace
                    * of closest center to each point instead of calling Partition(P, S) in the end.
                    */
                    if(!partition.containsKey(actPoint) || partition.get(actPoint)._2 > actDist )
                        partition.put(actPoint, new Tuple2<Vector, Double>(actCenter, actDist));

                    if ( actDist > dist ) {
                        dist = actDist;
                        max = p;
                    }
                }
            }
            Vector selected = points.remove(max);
            // Remove point p that maximizes d(p, S) from points and add to set of centers S
            S.add(selected);
            // Remove it also from partition since it becomes a new center and we don't need anymore to know its
            // distance from centers
            partition.remove(selected);
        }
        HashMap<Vector, ArrayList<Vector>> res = new HashMap<>();
        for (Vector c: S)
            res.put(c, new ArrayList<>());
        for(Map.Entry<Vector, Tuple2<Vector, Double>> point : partition.entrySet())
            res.get(point.getValue()._1).add(point.getKey());
        List<Tuple2<Vector, ArrayList<Vector>>> partitioned = new ArrayList<>();
        for(Map.Entry<Vector, ArrayList<Vector>> p : res.entrySet())
            partitioned.add(new Tuple2<>(p.getKey(), p.getValue()));
        return sc.parallelizePairs(partitioned);
    }

    public static Vector readPoint(String point) {
        return Vectors.dense(Arrays.stream(point.split(",")).mapToDouble(Double::parseDouble).toArray());
    }


    public static void runMapReduce(JavaRDD<Object> pointsRDD, int k, int L) {
        JavaRDD<Object> res = pointsRDD.mapPartitions(partition -> {
            return farthestFirstTraversal(partition, k);
        });
        System.out.println("rr");
        List<String> a = Arrays.asList("aa", "bb", "cc");
        Iterator<String> b = a.iterator();

    }

    public static void main(String[] args) throws IOException {
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: needed path, k, L.");
        }
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
