import torchvision.transforms as transforms

# sys.path.append('/Users/samchou/src/nrdg/AFQ-Insight/afqinsight/')
# from datasets import AFQDataset, download_hbn
# from ..datasets import *
from afqinsight.datasets import (
    AFQDataset,
    download_hbn,
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def load_hbn_dataset_as_pytorch(bundles_as_channels=True, channels_last=False):
    data_dir = download_hbn()

    afq_dataset = AFQDataset.from_files(
        fn_nodes=f"{data_dir}/nodes.csv",
        fn_subjects=f"{data_dir}/subjects.tsv",
        dwi_metrics=["dki_md", "dki_fa"],
        target_cols=["age", "sex", "scan_site_id"],
        label_encode_cols=["sex", "scan_site_id"],
        index_col="subject_id",
    )

    return afq_dataset.as_torch_dataset(
        bundles_as_channels=bundles_as_channels, channels_last=channels_last
    )


def load_hbn_dataset_as_tensorflow(bundles_as_channels=True, channels_last=True):
    data_dir = download_hbn()

    afq_dataset = AFQDataset.from_files(
        fn_nodes=f"{data_dir}/nodes.csv",
        fn_subjects=f"{data_dir}/subjects.tsv",
        dwi_metrics=["dki_md", "dki_fa"],
        target_cols=["age", "sex", "scan_site_id"],
        label_encode_cols=["sex", "scan_site_id"],
        index_col="subject_id",
    )

    return afq_dataset.as_tensorflow_dataset(
        bundles_as_channels=bundles_as_channels, channels_last=channels_last
    )


# if __name__ == "__main__":
#     hbn_pytorch_dataset = load_hbn_dataset_as_pytorch()
#     hbn_tensorflow_dataset = load_hbn_dataset_as_tensorflow()

#     print("PyTorch Dataset:", hbn_pytorch_dataset)
#     print("TensorFlow Dataset:", hbn_tensorflow_dataset)


# test_model = mlp4_pt(input_shape=784, n_classes=10)
# summary(test_model, input_size=(1, 784))

# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# test_tf_model = mlp4(input_shape=(784,), n_classes=10, verbose=True)

# #try to find gpu, if not use cpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #loading data in batches
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# model = mlp4_pt(input_shape=784, n_classes=10).to(device)
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(1):
#     model.train()
#     for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train"):
#         x, y = batch
#         x = x.view(-1, 28 * 28).to(device)
#         y = y.to(device)
#         optimizer.zero_grad()
#         yhat = model(x)
#         loss = criterion(yhat, y)
#         loss.backward()
#         optimizer.step()

#     model.eval()
#     with torch.no_grad():
#         test_loss = 0
#         correct = 0
#         for batch in tqdm(test_dataloader, desc=f"Epoch {epoch+1} Test"):
#             x, y = batch
#             x = x.view(-1, 784).to(device)
#             y = y.to(device)
#             yhat = model(x)
#             test_loss += criterion(yhat, y).item()
#             pred = yhat.argmax(dim=1, keepdim=True)
#             correct += pred.eq(y.view_as(pred)).sum().item()

#         test_loss /= len(test_dataloader.dataset)
#         accuracy = 100.0 * correct / len(test_dataloader.dataset)
#         print(f"Test loss: {test_loss}, Accuracy: {accuracy}%")
