#include "kapy-dialects.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"

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

static cl::opt<std::string> TensorStr("t",                       //
                                      cl::desc("Tensor String"), //
                                      cl::init(""),              //
                                      cl::cat(PrinterCategory));

LogicalResult printImpl(RankedTensorType tensorType, llvm::raw_ostream &os) {
  auto layout = kapy::getLayout<Attribute>(tensorType);
  auto &dialect = layout.getDialect();
  if (dialect.getNamespace() == "kgpu") {
    os << kapy::getLayoutString(tensorType);
    return success();
  }
  llvm::errs() << "Unsuported layout: " << layout << "\n";
  return failure();
}

LogicalResult printFromFile(MLIRContext *context, StringRef fileName,
                            RankedTensorType tensorType,
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

  auto printLambda = [&](StringRef name, Attribute encoding) {
    ss << encoding << "\n";
    ss << tensorType << "\n";
    tensorType = RankedTensorType::get(tensorType.getShape(),
                                       tensorType.getElementType(), encoding);
    return printImpl(tensorType, ss);
  };

  for (const auto &aliasDef : asmState.getAttributeAliasDefs())
    if (failed(printLambda(aliasDef.name, aliasDef.value)))
      return failure();
  return success();
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

  std::string string;
  llvm::raw_string_ostream ss(string);

  if (failed(printFromFile(&context, InputFileName, tensorType, ss)))
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
