load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "embed",
    srcs = ["embedding.cpp"],
    hdrs = ["embedding.h"],
    strip_include_prefix = "//embedding",
    visibility = ["//:__pkg__"],
)
