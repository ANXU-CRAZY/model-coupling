import com.supermap.data.*;
import com.supermap.data.conversion.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** Export rasters using an installed SuperMap SDK; the source is opened read-only. */
public final class ExportUdbx {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) throw new IllegalArgumentException("UDBX output_directory grid_name... or --all");
        Path input = Paths.get(args[0]).toAbsolutePath().normalize();
        Path out = Paths.get(args[1]).toAbsolutePath().normalize();
        if (!Files.isRegularFile(input)) throw new FileNotFoundException(input.toString());
        if (Files.exists(out)) throw new IOException("Refusing existing output directory: " + out);
        Set<String> selected = new HashSet<String>();
        for (int i = 2; i < args.length; i++) selected.add(args[i]);
        Workspace workspace = new Workspace();
        DatasourceConnectionInfo connection = new DatasourceConnectionInfo();
        connection.setEngineType(EngineType.UDBX);
        connection.setServer(input.toString());
        connection.setReadOnly(true);
        try {
            Datasource source = workspace.getDatasources().open(connection);
            if (source == null) throw new IOException("SuperMap did not open the datasource");
            Datasets datasets = source.getDatasets();
            List<Dataset> export = new ArrayList<Dataset>();
            Set<String> found = new HashSet<String>();
            for (int i = 0; i < datasets.getCount(); i++) {
                Dataset dataset = datasets.get(i);
                String name = dataset.getName();
                if (!(dataset instanceof DatasetGrid) && !(dataset instanceof DatasetImage)) continue;
                if (!selected.contains(name) && !selected.contains("--all")) continue;
                if (!name.matches("[A-Za-z0-9_]+")) throw new IOException("Unsafe output name: " + name);
                export.add(dataset);
                found.add(name);
            }
            if (export.isEmpty()) throw new IOException("No matching raster datasets");
            if (!selected.contains("--all") && !found.containsAll(selected))
                throw new IOException("Requested raster dataset does not exist");
            Files.createDirectories(out);
            for (Dataset dataset : export) {
                String name = dataset.getName();
                Path target = out.resolve(name + ".tif").normalize();
                if (!target.getParent().equals(out)) throw new IOException("Output escapes directory");
                Path projection = out.resolve(name + ".supermap_projection.xml");
                Files.write(projection, dataset.getPrjCoordSys().toXML().getBytes(StandardCharsets.UTF_8), StandardOpenOption.CREATE_NEW);
                ExportSettingTIF setting = new ExportSettingTIF(dataset, target.toString(), FileType.TIF);
                setting.setOverwrite(false);
                setting.setExportingPRJFile(true);
                setting.setExportingGeoTransformFile(true);
                DataExport exporter = new DataExport();
                try {
                    exporter.getExportSettings().add(setting);
                    ExportResult result = exporter.run();
                    if (result == null || result.getFailedSettings().length != 0 || result.getSucceedSettings().length != 1)
                        throw new IOException("Export failed for " + name);
                    System.out.println("EXPORTED " + target);
                } finally { exporter.dispose(); }
            }
            System.out.println("READ_ONLY_EXPORT_COMPLETED rasters=" + export.size());
        } finally {
            connection.dispose();
            workspace.dispose();
        }
    }
}
