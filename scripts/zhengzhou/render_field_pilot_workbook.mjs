import fs from 'node:fs/promises';
import path from 'node:path';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

// Run a local copy next to the bundled runtime node_modules junction.
// This workbook contains candidate coordinates and must remain local.
const inputDir = path.resolve(process.argv[2]);
const outputDir = path.resolve(process.argv[3]);
await fs.mkdir(outputDir, { recursive: true });
const outputFile = path.join(outputDir, '郑州先导核查与水鸟访问记录.xlsx');
try { await fs.access(outputFile); throw new Error('Output already exists'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const summary = JSON.parse(await fs.readFile(path.join(inputDir, 'summary.json'), 'utf8'));
const geo = JSON.parse(await fs.readFile(path.join(inputDir, 'landcover_verification_candidates_LOCAL_ONLY.geojson'), 'utf8'));
const workbook = Workbook.create();
const guide = workbook.worksheets.add('使用与定义');
const candidates = workbook.worksheets.add('地表核查点');
const visits = workbook.worksheets.add('水鸟访问');
const species = workbook.worksheets.add('物种记录');

function table(sheet, headers, rows, widths) {
  const count = Math.max(rows.length, 10);
  const values = [headers, ...rows, ...Array.from({length: count - rows.length}, () => headers.map(() => null))];
  sheet.getRangeByIndexes(0, 0, values.length, headers.length).values = values;
  const all = sheet.getRangeByIndexes(0, 0, values.length, headers.length);
  all.format.font = {name: 'Arial', size: 10, color: '#202A33'};
  all.format.verticalAlignment = 'center';
  all.format.rowHeight = 24;
  const header = sheet.getRangeByIndexes(0, 0, 1, headers.length);
  header.format.fill = '#284B63';
  header.format.font = {name: 'Arial', size: 10, bold: true, color: '#FFFFFF'};
  header.format.horizontalAlignment = 'center';
  header.format.wrapText = true;
  header.format.rowHeight = 38;
  widths.forEach((width, i) => sheet.getRangeByIndexes(0, i, values.length, 1).format.columnWidth = width);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
}

guide.showGridLines = false;
guide.getRange('A2').values = [['郑州市先导核查与水鸟访问记录']];
guide.getRange('A2').format.font = {name: 'Arial', size: 14, bold: true};
const instructions = [
  ['任务', '记录规则'],
  ['地表核查', '36个初选点、36个备用点。先检查可达性与实际地表，留下日期、照片和复核证据。'],
  ['定位', '候选经纬度是100m像元中心，可能位于水面。实际观测站位另行记录，不能直接导航入水。'],
  ['替换', '使用备用点必须记录原点、原因和顺序；不能因未见鸟换点。'],
  ['访问单元', '一行表示同一调查范围的一次有效访问。范围ID需关联实际调查多边形或轨迹文件。'],
  ['主要响应', '完整调查范围内目标水鸟群落的检出/未检出；物种范围采用经复核的目标名录。'],
  ['零与缺失', '完成完整目标清单且没有水鸟时填“未检出”，记录计数0；未调查、范围不完整或不确定留空。'],
  ['努力量', '记录实际有效分钟、人数及距离。开始结束时间跨度不能替代有效搜索时间。'],
  ['空间支撑', '调查范围可以跨多个像元；不得将全部鸟数自动归入候选点的单一土地类别。'],
  ['重复访问', '建议起点：20分钟静点、每季3次、7天内完成。先导评估后确定正式协议和样本量。'],
  ['检测假设', '上述重复窗口不是已验证的占域闭合条件。迁徙水鸟流动需另行评估。'],
  ['物种记录', '每行一物种×访问。保留鉴定证据、数量及未知数量标志，不强行填0。'],
  ['坐标与来源', '访问采用WGS84并记录定位精度。当前市域候选边界基准仍待独立验证。'],
  ['当前数据', '历史8407个访问候选无已确认完整清单或努力量，不作为独立生境校准标签。'],
  ['类别编码', '当前1–9类仍未恢复正式名称；历史0–8表含snow，不能直接套用。'],
  ['参考方法', 'https://ebird.github.io/ebird-best-practices/ebird.html'],
  ['采样范围', `${summary.pilot_sampling.sampling_frame_pixels.toLocaleString('en-US')}个有效像元；只覆盖候选市域中LULC/压力有效区。`],
];
guide.getRange(`A4:B${3 + instructions.length}`).values = instructions;
guide.getRange('A4:B21').format.font = {name: 'Arial', size: 10};
guide.getRange('A4:B4').format.fill = '#284B63';
guide.getRange('A4:B4').format.font.color = '#FFFFFF';
guide.getRange('A4:B4').format.font.bold = true;
guide.getRange('A4:B21').format.verticalAlignment = 'center';
guide.getRange('A4:A21').format.columnWidth = 17;
guide.getRange('B4:B21').format.columnWidth = 104;
guide.getRange('B5:B21').format.wrapText = true;
guide.getRange('A4:B21').format.rowHeight = 33;

const pointRows = geo.features.map(f => [f.properties.candidate_id, f.properties.lucode,
  f.properties.pressure_stratum === 'low' ? '较低' : '较高',
  f.properties.selection_status === 'primary' ? '初选' : '备用',
  f.geometry.coordinates[0], f.geometry.coordinates[1], null, null, null, null, null, null]);
table(candidates, ['候选点ID','栅格代码','人类活动层','选点序列','经度WGS84','纬度WGS84','可达性','核查日期','实际地表','照片编号','复核状态','替换原因'],
  pointRows, [18,12,15,12,17,17,14,15,22,20,16,34]);
candidates.getRange('E2:F73').setNumberFormat('0.000000');
candidates.getRange('B2:B73').setNumberFormat('0');
candidates.getRange('H2:H73').setNumberFormat('yyyy-mm-dd');
candidates.getRange('G2:L73').format.fill = '#FFF4CD';
candidates.getRange('G2:G73').dataValidation = {rule:{type:'list',values:['可达','不可达','待核实']}};
candidates.getRange('K2:K73').dataValidation = {rule:{type:'list',values:['待复核','已复核','不一致']}};

const visitHeaders = ['访问ID','调查单元ID','候选点ID','日期','开始时刻','结束时刻','有效分钟','人数','协议','完整目标清单','调查范围ID','实际站位经度WGS84','实际站位纬度WGS84','定位精度m','距离km','水位或深度m','水位参照','天气','目标名录版本','目标水鸟检出','计数总和','计数完整性','可达性','未完成原因','重复访问组ID','重复序号','记录人','证据文件','备注'];
table(visits, visitHeaders, [], [20,20,18,15,14,14,13,10,16,18,23,24,24,15,13,17,20,24,22,18,14,17,14,26,23,13,15,25,32]);
visits.getRange('A2:AC11').format.fill = '#FFF4CD';
visits.getRange('D2:D11').setNumberFormat('yyyy-mm-dd');
visits.getRange('E2:F11').setNumberFormat('hh:mm');
visits.getRange('L2:M11').setNumberFormat('0.000000');
visits.getRange('O2:P11').setNumberFormat('0.00');
visits.getRange('J2:J11').dataValidation = {rule:{type:'list',values:['是','否','未知']}};
visits.getRange('T2:T11').dataValidation = {rule:{type:'list',values:['检出','未检出','未知']}};
visits.getRange('V2:V11').dataValidation = {rule:{type:'list',values:['完整计数','部分数量未知','未完成']}};
visits.getRange('W2:W11').dataValidation = {rule:{type:'list',values:['可达','不可达','待核实']}};

table(species, ['访问ID','物种中文名','拉丁名','数量','数量状态','鉴定状态','目标名录版本','证据文件','备注'],
  [], [20,22,30,12,18,20,22,28,36]);
species.getRange('A2:I11').format.fill = '#FFF4CD';
species.getRange('E2:E11').dataValidation = {rule:{type:'list',values:['准确记录','估计','数量未知']}};
species.getRange('F2:F11').dataValidation = {rule:{type:'list',values:['已复核','待复核','无法鉴定']}};

workbook.recalculate();
const inspection = await workbook.inspect({kind:'table',range:'地表核查点!A1:F5',include:'values,formulas',tableMaxRows:5,tableMaxCols:6,maxChars:2500});
await fs.writeFile(path.join(outputDir, 'workbook_inspection.ndjson'), inspection.ndjson);
const errors = await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:20},maxChars:2000});
await fs.writeFile(path.join(outputDir, 'workbook_error_scan.ndjson'), errors.ndjson);
for (const [name, range] of [['使用与定义','A1:B21'],['地表核查点','A1:G8'],['水鸟访问','A1:K8'],['物种记录','A1:I8']]) {
  const preview = await workbook.render({sheetName:name,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(outputDir, `preview_${name}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputFile);
console.log(JSON.stringify({output:outputFile,sheets:4,candidates:pointRows.length,observed_visits_created:0}));
