idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES freertos
)

# Flags de compilação otimizadas
target_compile_options(${COMPONENT_LIB} PRIVATE -std=c++14 -Wall -Wextra)

