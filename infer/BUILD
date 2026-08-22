load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

sh_binary(
    name = "infer",
    srcs = ["run.sh"],
    data = [
        ":infer_bin",
        "model/gen_tokens.py",
        "model/input.txt",
    ],
)

cc_binary(
    name = "infer_bin",
    srcs = [
        "main.cpp",
        "main.h",
    ],
    data = [
        "model/gpt2.bin",
        "model/tokens.bin",
        "model/vocab.bin",
    ],
    deps = [
        "//embedding:embed",
        "//transformer/attention:attention",
        "//transformer/layer_norm:layer_norm",
        "//transformer/mlp:mlp",
    ],
)
