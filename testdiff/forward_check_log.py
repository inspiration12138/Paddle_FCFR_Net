from reprod_log import ReprodDiffHelper
import numpy as np

if __name__ == '__main__':
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("../result/forward_ref.npy")
    paddle_info = diff_helper.load_info("../result/forward_paddle.npy")

    print(torch_info)
    print(paddle_info)

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="../result/log/forward_diff.log")
