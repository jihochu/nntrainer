#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file genModelsMultiout_v2.py
# @date 24 November 2021
# @brief Generate multiout model tcs
# @author Jihoon lee <jhoon.it.lee@samsung.com>

from recorder_v2 import record_v2, inspect_file
import torch


class SplitAndJoin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)
        self.fc1 = torch.nn.Linear(1, 3)
        self.fc2 = torch.nn.Linear(1, 3)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        a0, a1 = torch.split(out, 1, dim=1)
        out = self.fc1(a0) + self.fc2(a1)
        loss = self.loss(out, labels[0])
        return out, loss


class SplitAndJoinDangle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 4)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(1, 3)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        #         input
        #        (split)
        #   a0   a1   a2   a3
        #   |               |
        #   b0             b1
        #         (add)
        #          c0
        #
        # only c0 is fed to loss, a1 and a2 is dangled
        a0, a1, a2, a3 = torch.split(out, 1, dim=1)
        a0 = self.sigmoid(a0)
        a1 = self.sigmoid(a1)
        a2 = self.sigmoid(a2)
        a3 = self.sigmoid(a3)

        b0 = self.fc1(a0)
        b3 = self.fc1(a3) # shared
        c0 = b0 + b3
        out = self.sigmoid(c0)
        loss = self.loss(out, labels[0])
        return out, loss


class OneToOne(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        a0, a1 = torch.split(out, 1, dim=1)
        out = a0 + a1
        loss = self.loss(out, labels[0])
        return out, loss


class OneToMany(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 3)
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        out = self.fc(inputs[0])
        a0, a1, a2 = torch.split(out, 1, dim=1)
        b0 = a0 + a1
        c0 = a0 + a1
        d0 = b0 + c0 + a2
        loss = self.loss(d0, labels[0])
        return d0, loss


if __name__ == "__main__":
    record_v2(
        SplitAndJoin(),
        iteration=2,
        input_dims=[(5, 3)],
        label_dims=[(5, 3)],
        name="split_and_join"
    )

    record_v2(
        SplitAndJoinDangle(),
        iteration=3,
        input_dims=[(5, 3)],
        label_dims=[(5, 3)],
        name="split_and_join_dangle"
    )

    record_v2(
        OneToOne(),
        iteration=2,
        input_dims=[(5, 3)],
        label_dims=[(5, 1)],
        name="one_to_one"
    )

    record_v2(
        OneToMany(),
        iteration=2,
        input_dims=[(5, 2)],
        label_dims=[(5, 1)],
        name="one_to_many"
    )

#    inspect_file("split_and_join_dangle.nnmodelgolden")
