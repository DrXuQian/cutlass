/*
 * Stub implementation for TllmException
 * Simplified version for standalone MoE kernel extraction
 */

#include "tllmException.h"
#include <cstdarg>
#include <cstring>

namespace tensorrt_llm::common
{

// fmtstr_ implementation
char const* fmtstr_(char const* format, char* (*allocator)(void*, size_t), void* ctx, va_list* args)
{
    va_list args_copy;
    va_copy(args_copy, *args);

    // Get required size
    int size = vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);

    if (size < 0) {
        return "format error";
    }

    char* buffer = allocator(ctx, size + 1);
    if (!buffer) {
        return "allocation error";
    }

    vsnprintf(buffer, size + 1, format, *args);
    return buffer;
}

TllmException::TllmException(char const* file, std::size_t line, char const* msg)
    : std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + msg)
    , mNbFrames(0)
{
    // Skip backtrace for simplicity
}

TllmException::~TllmException() noexcept = default;

std::string TllmException::getTrace() const
{
    return ""; // Skip backtrace for simplicity
}

std::string TllmException::demangle(char const* name)
{
    return name; // Skip demangling for simplicity
}

RequestSpecificException::RequestSpecificException(
    std::string const& file, std::size_t line, char const* msg, uint64_t requestID, RequestErrorCode errorCode)
    : std::runtime_error(file + ":" + std::to_string(line) + " " + msg)
    , mRequestID(requestID)
    , mErrorCode(errorCode)
{
}

RequestSpecificException::~RequestSpecificException() noexcept = default;

uint64_t RequestSpecificException::getRequestId() const noexcept
{
    return mRequestID;
}

RequestErrorCode RequestSpecificException::getErrorCode() const noexcept
{
    return mErrorCode;
}

} // namespace tensorrt_llm::common
