package person.me.galaxy.lambda;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import person.me.galaxy.lambda.entity.Album;

import java.util.ArrayList;
import java.util.List;
import java.util.function.ToLongFunction;

public class LambdaDebug {
    private static final Logger logger = LoggerFactory.getLogger(LambdaDebug.class);
    List<Album> albums = new ArrayList<>();

    public long countTracks() {
        return albums.stream()
            .mapToLong(album -> (long) album.getTracks().size())
            .sum();
    }

    public long countMusicians() {
        return albums.stream()
                .mapToLong(album -> album.getMusicians().stream().count())
                .sum();
    }

    public long countRunningTime() {
        return albums.stream()
                .mapToLong(album -> album.getTracks().stream()
                        .mapToLong(track -> track.getLength()).sum())
                .sum();
    }


    public long countFeature(ToLongFunction<Album> function) {
        return albums.stream().mapToLong(function).sum();
    }

    public long countTracks2() {
        return countFeature(album -> album.getTracks().stream().count());
    }

    public long countRunningTime2() {
        return countFeature(album -> album.getTracks().stream()
                .mapToLong(track -> track.getLength())
                .sum());
    }
    public long countMusicians2() {
        return countFeature(album -> album.getMusicians().stream().count());
    }


    public static void main(String[] args) {
        logger.debug("Look at this: " + "expensiveOperation()");

        if (logger.isDebugEnabled()) {
            logger.debug("Look at this: " + "expensiveOperation()");
        }

//        logger.debug(() -> "Look at this: " + "expensiveOperation()3");
    }
}
