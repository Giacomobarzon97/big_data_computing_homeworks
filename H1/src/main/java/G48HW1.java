import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class G48HW1 {
  public static void main(String[] args) throws IOException {
    if (args.length != 2) {
      throw new IllegalArgumentException("USAGE: num_partitions file_path");
    }
    SparkConf conf = new SparkConf(true).setAppName("Homework1");
    JavaSparkContext sc = new JavaSparkContext(conf);
    sc.setLogLevel("WARN");
    // INPUT READING
    // Read number of partitions
    int K = Integer.parseInt(args[0]);
    // Read input file and subdivide it into K random partitions
    JavaRDD<String> pairString = sc.textFile(args[1]).repartition(K);

    // SETTING GLOBAL VARIABLES
    long numpairs,numclasses;
    numpairs = pairString.count();
    System.out.println("Number of pairs = " + numpairs);
    JavaPairRDD<String, Long> count;


    //First Exercise
    System.out.println("VERSION WITH DETERMINISTIC PARTITIONS\n");
    Random randomGenerator = new Random();
    count = pairString
            .flatMapToPair((pair) -> {    // <-- MAP PHASE (R1)
              int index=Integer.parseInt(pair.split(" ")[0]);
              String category=pair.split(" ")[1];

              ArrayList<Tuple2<Integer, String>> pairs = new ArrayList<>();
              pairs.add(new Tuple2<>(index%K,category));
              return pairs.iterator();
            })
            .groupByKey()    // <-- REDUCE PHASE (R1)
            .flatMapToPair((pair) -> {
              HashMap<String, Long> counts = new HashMap<>();
              for (String  s: pair._2()) {
                counts.put(s,counts.getOrDefault(s,0L)+1);
              }
              ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
              for (Map.Entry<String, Long> e : counts.entrySet()) {
                pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
              }
              return pairs.iterator();
            }).groupByKey()// <-- REDUCE PHASE (R2)
            .flatMapToPair((pair) -> {
                Long sum=0L;
                for (Long  i: pair._2()) {
                    sum=sum+i;
                }
                ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                pairs.add(new Tuple2<String,Long>(pair._1(),sum));
                return pairs.iterator();
            });
      String res="Output pairs = ";
      for(Tuple2<String,Long> tuple:count.sortByKey().collect()) {
          res=res+"("+tuple._1()+","+tuple._2()+"),";
      }
      System.out.println(res.substring(0,res.length()-1)+";");
      
  }

}
