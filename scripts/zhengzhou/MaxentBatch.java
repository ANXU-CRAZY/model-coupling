import java.io.*;
import java.lang.reflect.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;

/** Serial official MaxEnt jobs in a JVM, with a fresh isolated loader per job.
 * TSV rows: job index, then one complete MaxEnt argument per column.
 * No density class is loaded through the application's parent class loader.
 * Use threads=1. The Python adapter validates inputs, outputs and manifests.
 */
public final class MaxentBatch {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("jar task_tsv");
        URL jar = Paths.get(args[0]).toAbsolutePath().toUri().toURL();
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(args[1]), StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) continue;
                String[] fields = line.split("\t", -1);
                if (fields.length < 2) throw new IllegalArgumentException("Empty task");
                String id = fields[0];
                System.out.println("MAXENT_BATCH_START\t" + id + "\t" + System.currentTimeMillis());
                System.out.flush();
                String[] parameters = java.util.Arrays.copyOfRange(fields, 1, fields.length);
                int priority = Thread.currentThread().getPriority();
                ClassLoader previous = Thread.currentThread().getContextClassLoader();
                try (URLClassLoader loader = new URLClassLoader(new URL[]{jar}, ClassLoader.getPlatformClassLoader())) {
                    Thread.currentThread().setContextClassLoader(loader);
                    Class<?> paramsClass = Class.forName("density.Params", true, loader);
                    Object params = paramsClass.getConstructor().newInstance();
                    Object errors = paramsClass.getMethod("readFromArgs", String[].class).invoke(params, (Object)parameters);
                    if (errors != null) throw new IllegalArgumentException("Unknown arguments: " + errors);
                    paramsClass.getMethod("setSelections").invoke(params);
                    Class<?> runnerClass = Class.forName("density.Runner", true, loader);
                    Object runner = runnerClass.getConstructor(paramsClass).newInstance(params);
                    try {
                        runnerClass.getMethod("start").invoke(runner);
                        Field interrupted = Class.forName("density.Utils", true, loader).getDeclaredField("interrupt");
                        interrupted.setAccessible(true);
                        if (interrupted.getBoolean(null)) throw new IllegalStateException("MaxEnt interrupted");
                    } finally {
                        runnerClass.getMethod("end").invoke(runner);
                    }
                    System.out.println("MAXENT_BATCH_OK\t" + id + "\t" + System.currentTimeMillis());
                } catch (Throwable failure) {
                    Throwable cause = failure instanceof InvocationTargetException ? failure.getCause() : failure;
                    System.err.println("MAXENT_BATCH_FAILED\t" + id + "\t" + cause);
                    cause.printStackTrace(System.err);
                    throw new RuntimeException("Fail closed at task " + id, cause);
                } finally {
                    Thread.currentThread().setContextClassLoader(previous);
                    Thread.currentThread().setPriority(priority);
                }
                System.out.flush();
                System.gc();
            }
        }
    }
}
