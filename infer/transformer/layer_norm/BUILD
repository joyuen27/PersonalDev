load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "layer_norm",
    srcs = ["layer_norm.cpp"],
    hdrs = ["layer_norm.h"],
    strip_include_prefix = "//transformer/layer_norm",
    visibility = ["//:__pkg__"],
)
