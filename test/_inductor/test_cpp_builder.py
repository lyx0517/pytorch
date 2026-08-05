from unittest import TestCase, main, mock

import torch._inductor.cpp_builder as torch_cpp_builder

from torch_npu._inductor import cpp_builder as npu_cpp_builder


class TestCppBuilder(TestCase):
    def test_register_npu_cpp_device_options(self):
        original_get_options = torch_cpp_builder.get_cpp_torch_device_options

        with mock.patch.object(
            torch_cpp_builder, "register_cpp_device_options"
        ) as register_options:
            npu_cpp_builder.register_npu_cpp_device_options()

        register_options.assert_called_once_with(
            "npu", npu_cpp_builder.get_cpp_torch_device_options
        )
        self.assertIs(
            torch_cpp_builder.get_cpp_torch_device_options,
            original_get_options,
        )


if __name__ == "__main__":
    main()
