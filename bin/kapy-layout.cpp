#include "kapy-dialects.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include <llvm/Support/SMLoc.h>

using namespace llvm;
using namespace mlir;

cl::OptionCategory PrinterCategory("Available Print Options",
                                   "Options for the layout printing");

static cl::opt<std::string> InputFileName("i", //
                                          cl::desc("Input File Name"),
                                          cl::init(""),
                                          cl::cat(PrinterCategory));

static cl::opt<std::string> WriteFileName("o", //
                                          cl::desc("Write File Name"),
                                          cl::init(""),
                                          cl::cat(PrinterCategory));

static cl::opt<std::string> LayoutStr("l",                       //
                                      cl::desc("Layout String"), //
                                      cl::init(""),              //
                                      cl::cat(PrinterCategory));

static cl::opt<std::string> TensorStr("t",                       //
                                      cl::desc("Tensor String"), //
                                      cl::init(""),              //
                                      cl::cat(PrinterCategory));

static cl::opt<int> NumWarps("w",                                     //
                             cl::desc("Number of warps in each CTA"), //
                             cl::init(4),                             //
                             cl::cat(PrinterCategory));

LogicalResult printImpl(RankedTensorType tensorType, int numWarps,
                        llvm::raw_ostream &os) {
  auto dialectName = tensorType.getEncoding().getDialect().getNamespace();
  if (dialectName == "kgpu") {
    os << kapy::getLayoutString(tensorType, numWarps);
    return success();
  }
  llvm::errs() << "Unsuported layout: " << tensorType.getEncoding() << "\n";
  return failure();
}

LogicalResult printFromFile(MLIRContext *context, StringRef fileName,
                            RankedTensorType tensorType, int numWarps,
                            llvm::raw_string_ostream &ss) {
  if (fileName.empty())
    return success();

  auto fileOrError = MemoryBuffer::getFileOrSTDIN(fileName);
  if (auto errorCode = fileOrError.getError()) {
    llvm::errs() << "Could not open input file: " << errorCode.message()
                 << "\n";
    return failure();
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrError), SMLoc());
  ParserConfig config(context);
  auto asmState = AsmParserState();

  Block parsedIR;
  if (failed(parseAsmSourceFile(sourceMgr, &parsedIR, config, &asmState))) {
    llvm::errs() << "Failed to parse the input file: " << fileName << "\n";
    return failure();
  }

  auto printLambda = [&](StringRef name, Attribute layout) {
    ss << "Print layout: #" << name << " = " << layout << "\n";
    return printImpl(kapy::cloneWith(tensorType, layout), numWarps, ss);
  };

  for (const auto &aliasDef : asmState.getAttributeAliasDefs())
    if (failed(printLambda(aliasDef.name, aliasDef.value)))
      return failure();
  return success();
}

LogicalResult printFromString(MLIRContext *context, StringRef layoutStr,
                              RankedTensorType tensorType, int numWarps,
                              llvm::raw_string_ostream &ss) {
  if (layoutStr.empty())
    return success();

  auto layout = parseAttribute(layoutStr, context);
  if (!layout) {
    llvm::errs() << "Invalid layout: " << layoutStr << "\n";
    return failure();
  }

  ss << "Print layout: " << layout << "\n";
  return printImpl(kapy::cloneWith(tensorType, layout), numWarps, ss);
}

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(PrinterCategory);
  cl::ParseCommandLineOptions(argc, argv, "Kapy layout printer\n");

  DialectRegistry registry;
  registerAllKapyDialects(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  if (TensorStr.empty()) {
    llvm::errs() << "Must specify the tensor string argument\n";
    return 1;
  }

  auto parsedType = parseType(TensorStr, &context);
  if (!parsedType) {
    llvm::errs() << "Failed to parse the tensor string: " << TensorStr;
    return 1;
  }
  auto tensorType = dyn_cast<RankedTensorType>(parsedType);
  if (!tensorType) {
    llvm::errs() << "Invalid tensor string: " << TensorStr << "\n";
    return 1;
  }

  std::string str;
  llvm::raw_string_ostream ss(str);

  if (failed(printFromFile(&context, InputFileName, tensorType, NumWarps, ss)))
    return 1;

  if (failed(printFromString(&context, LayoutStr, tensorType, NumWarps, ss)))
    return 1;

  if (WriteFileName.empty()) {
    llvm::outs() << ss.str();
  } else {
    std::error_code errorCode;
    llvm::raw_fd_ostream writeFile(WriteFileName, errorCode,
                                   llvm::sys::fs::OF_Text);
    if (errorCode) {
      llvm::errs() << "Error: " << errorCode.message() << " : unable to open "
                   << WriteFileName << " to write\n";
      return 1;
    }
    writeFile << ss.str();
    writeFile.close();
  }
  return 0;
}
