package person.me.galaxy.ml.word2vec;

import java.io.File;
import java.io.IOException;

//import com.ansj.vec.domain.Term;
//import org.ansj.splitWord.analysis.ToAnalysis;
//import com.alibaba.fastjson.JSONObject;


public class TestWord2vec {
    private static final String fileName = "/swresult_withoutnature.txt";
    private static String path = TestWord2vec.class.getResource(fileName).getPath();


    public static void main(String[] args) throws IOException {
//        File[] files = new File("corpus/sport/").listFiles();
        System.out.println(fileName);
        System.out.println(path);
        //构建语料
//        try (FileOutputStream fos = new FileOutputStream(sportCorpusFile)) {
//            assert files != null;
//            for (File file : files) {
//                if (file.canRead() && file.getName().endsWith(".txt")) {
//                    parseFile(fos, file);
//                }
//            }
//        }
        String savePath = "vector.mod";
//        //进行分词训练
        Learn lean = new Learn();

        final File sportCorpusFile = new File(path);
        lean.learnFile(sportCorpusFile);
        lean.saveModel(new File(savePath));


        //加载测试
        Word2VEC w2v = new Word2VEC();
        w2v.loadJavaModel(savePath);
        System.out.println(w2v.distance("新年"));
    }


//    private static void parseFile(FileOutputStream fos, File file) throws FileNotFoundException, IOException {
//        // TODO Auto-generated method stub
//        try (BufferedReader br = IOUtil.getReader(file.getAbsolutePath(), IOUtil.UTF8)) {
//            String temp = null;
//            JSONObject parse = null;
//            while ((temp = br.readLine()) != null) {
//                parse = JSONObject.parseObject(temp);
//                parseStr(fos, parse.getString("title"));
//                parseStr(fos, StringUtil.rmHtmlTag(parse.getString("content")));
//            }
//        }
//    }
//
//    private static void parseStr(FileOutputStream fos, String title) throws IOException {
//        List<Term> parse2 = ToAnalysis.parse(title);
//        StringBuilder sb = new StringBuilder();
//        for (Term term : parse2) {
//            sb.append(term.getName());
//            sb.append(" ");
//        }
//        fos.write(sb.toString().getBytes());
//        fos.write("\n".getBytes());
//    }
}
