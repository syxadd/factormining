
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt

"""
使用backtrader进行回测交易
"""

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())