import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class G48HW2 {
    private static ArrayList<Vector>readCSV(String path) throws FileNotFoundException {
        ArrayList<Vector> records=new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(path));) {
            while (scanner.hasNext()) {
                double[]  values = Arrays.stream(scanner.next().split(","))
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                records.add(Vectors.dense(values));
            }
        }
        return records;
    }

    private static double exactMPD(ArrayList<Vector> s){
        double max_distance=-1;
        for(int i=0; i<s.size();i++){
            for(int j=i+1; j<s.size();j++){
                double current_distance=Vectors.sqdist(s.get(j),s.get(i));
                if (current_distance>max_distance) {
                    max_distance=current_distance;
                }
            }
        }
        return max_distance;
    }

    public static void main(String[] args) throws IOException {
        //final String dir = System.getProperty("user.dir");
        if (args.length != 1) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");
        // Read number of partitions
        String path = args[0];
        ArrayList<Vector> inputPoints = readCSV(path);
        double exact_distance=exactMPD(inputPoints);
        System.out.println(exact_distance);
    }
}
