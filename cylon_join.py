from pycylon import CylonContext, Table
import pyarrow as pa
import pyarrow.parquet as pq
import pandas
import time

if __name__ == "__main__":
	t1 = pq.read_table('table_1.parquet')
	t2 = pq.read_table('table_2.parquet')

	s = time.time()
	ctx = CylonContext()

	ct1 = Table(t1,ctx)
	ct2 = Table(t2,ctx)

	joined = ct1.join(ct2, 'inner', 'hash', on=['run', 'event', 'luminosityBlock'])
	print(len(joined.to_arrow().schema.names))
	e = time.time()
	print(e-s)

	s = time.time()
	df1 = t1.to_pandas(split_blocks=True, self_destruct=True)
	df2 = t2.to_pandas(split_blocks=True, self_destruct=True)

	rdf = df1.merge(df2, how='inner', on=['run', 'luminosityBlock', 'event'])
	rt = pa.Table.from_pandas(rdf)
	print(len(rt.schema.names))
	e = time.time()
	print(e-s)
