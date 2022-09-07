// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

// This example showcases various ways to work with Datasets. It's
// intended to be paired with the documentation.

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/exec/exec_plan.h>
#include <arrow/compute/exec/expression.h>
#include <arrow/csv/api.h>

#include <arrow/dataset/api.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/plan.h>
#include <arrow/dataset/scanner.h>

#include <arrow/io/api.h>
#include <arrow/io/interfaces.h>
#include <arrow/io/memory.h>
#include <arrow/io/stdio.h>

#include <arrow/ipc/api.h>

#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/filesystem.h>

#include <arrow/result.h>
#include <arrow/status.h>

#include <arrow/util/vector.h>

#include <parquet/arrow/reader.h>

#include <iostream>
#include <vector>

namespace ds = arrow::dataset;
namespace cp = arrow::compute;


arrow::Result<std::shared_ptr<arrow::dataset::Dataset>> ReadParquetFile(std::string path) {
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    auto format = std::make_shared<arrow::dataset::ParquetFileFormat>();
    arrow::fs::LocalFileSystem fs;
    std::shared_ptr<arrow::io::RandomAccessFile> input = fs.OpenInputFile(path).ValueOrDie();

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    arrow::Status st = parquet::arrow::OpenFile(input, pool, &arrow_reader);
    
    std::shared_ptr<arrow::Table> table;
    st = arrow_reader->ReadTable(&table);

    auto ds = std::make_shared<arrow::dataset::InMemoryDataset>(table);
    arrow::Result<std::shared_ptr<arrow::dataset::InMemoryDataset>> result(std::move(ds));
    return result;
}


arrow::Status DoHashJoin() {
  cp::ExecContext exec_context;

  arrow::dataset::internal::Initialize();

  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<cp::ExecPlan> plan,
                        cp::ExecPlan::Make(&exec_context));

  arrow::AsyncGenerator<arrow::util::optional<cp::ExecBatch>> sink_gen;

  cp::ExecNode* left_source;
  cp::ExecNode* right_source;

  ARROW_ASSIGN_OR_RAISE(auto l_dataset, ReadParquetFile("table_1.parquet"));
  ARROW_ASSIGN_OR_RAISE(auto r_dataset, ReadParquetFile("table_2.parquet"));

  auto l_options = std::make_shared<arrow::dataset::ScanOptions>();
  // create empty projection: "default" projection where each field is mapped to a
  // field_ref
  l_options->projection = cp::project({}, {});

  auto r_options = std::make_shared<arrow::dataset::ScanOptions>();
  // create empty projection: "default" projection where each field is mapped to a
  // field_ref
  r_options->projection = cp::project({}, {});

  // construct the scan node
  auto l_scan_node_options = arrow::dataset::ScanNodeOptions{l_dataset, l_options};
  auto r_scan_node_options = arrow::dataset::ScanNodeOptions{r_dataset, r_options};

  ARROW_ASSIGN_OR_RAISE(left_source,
                        cp::MakeExecNode("scan", plan.get(), {}, l_scan_node_options));
  ARROW_ASSIGN_OR_RAISE(right_source,
                        cp::MakeExecNode("scan", plan.get(), {}, r_scan_node_options));

  arrow::compute::HashJoinNodeOptions join_opts{arrow::compute::JoinType::INNER,
                                                /*in_left_keys=*/{"run", "luminosityBlock", "event"},
                                                /*in_right_keys=*/{"run", "luminosityBlock", "event"},
                                                /*filter*/ arrow::compute::literal(true),
                                                /*output_suffix_for_left*/ "_l",
                                                /*output_suffix_for_right*/ "_r"};

  ARROW_ASSIGN_OR_RAISE(
      auto hashjoin,
      cp::MakeExecNode("hashjoin", plan.get(), {left_source, right_source}, join_opts));

  ARROW_ASSIGN_OR_RAISE(std::ignore, cp::MakeExecNode("sink", plan.get(), {hashjoin},
                                                      cp::SinkNodeOptions{&sink_gen}));
  // expected columns l_a, l_b
  std::shared_ptr<arrow::RecordBatchReader> sink_reader = cp::MakeGeneratorReader(
      hashjoin->output_schema(), std::move(sink_gen), exec_context.memory_pool());

  // validate the ExecPlan
  ARROW_RETURN_NOT_OK(plan->Validate());
  // start the ExecPlan
  ARROW_RETURN_NOT_OK(plan->StartProducing());

  // collect sink_reader into a Table
  std::shared_ptr<arrow::Table> response_table;

  ARROW_ASSIGN_OR_RAISE(response_table,
                        arrow::Table::FromRecordBatchReader(sink_reader.get()));

  std::cout << "Results : " << response_table->ToString() << std::endl;

  return arrow::Status::OK();
}

int main(int argc, char** argv) {
  auto status = DoHashJoin();
  if (!status.ok()) {
    std::cerr << "Error occurred: " << status.message() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}