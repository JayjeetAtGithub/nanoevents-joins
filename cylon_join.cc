/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <glog/logging.h>
#include <chrono>

#include <cylon/net/mpi/mpi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>
#include <cylon/io/arrow_io.hpp>

int main(int argc, char *argv[]) {
  if (argc < 4) {
    LOG(ERROR) << "There should be two arguments with paths to parquet files";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  std::shared_ptr<cylon::CylonContext> ctx = cylon::CylonContext::Init();
  std::cout << "Initialized table variables\n";
  std::shared_ptr<arrow::Table> first_table, second_table, joined;

  std::cout << "Reading table 1\n";
  first_table = cylon::io::ReadParquet(ctx, argv[1]).ValueOrDie();
  std::cout << first_table->schema()->ToString() << std::endl;

  std::cout << "\n\n\n\n\n\n\n";

  std::cout << "Reading table 2\n";
  second_table = cylon::io::ReadParquet(ctx, argv[2]).ValueOrDie();
  std::cout << second_table->schema()->ToString() << std::endl;

  auto read_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_start).count() << "[ms]";

  auto join_config = cylon::join::config::JoinConfig(cylon::join::config::JoinType::INNER,
                                                     {487, 488, 489},
                                                     {487, 488, 489},
                                                     cylon::join::config::JoinAlgorithm::HASH,
                                                     "l_",
                                                     "r_");

  std::cout << second_table->GetColumnByName(argv[3])->ToString() << std::endl;

  auto status = cylon::join::JoinTables(first_table, second_table, join_config, &joined);

  if (!status.is_ok()) {
    LOG(INFO) << "Table join failed ";
    ctx->Finalize();
    return 1;
  }
  auto join_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "First table had : " << first_table->num_rows() << " and Second table had : "
            << second_table->num_rows() << ", Joined has : " << joined->num_rows();
  LOG(INFO) << "Join done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                join_end_time - read_end_time).count() << "[ms]";

  std::cout << joined->schema()->field_names().size() << std::endl;
  std::cout << joined->GetColumnByName(argv[3])->ToString() << std::endl;

  ctx->Finalize();
  return 0;
}
